#!/usr/bin/env python3
"""Offline MiniMax-H3 Turbo LoRA merger for the FL2VA transformer shards.

This tool is deliberately conservative: it never writes into the official base
model tree, validates the mixed-rank LoRA/base mapping from safetensors headers,
then creates a separate sharded transformer checkpoint where each targeted base
weight is replaced with::

    W_eff = cast_to_base_dtype(float32(W) + strength * float32(B @ A))

The default math contract is strength=1.0 with BF16 output.  A pure-Python engine
is included for tiny fixtures and static CI.  Real 13-shard checkpoint execution
should use an environment with torch (and, for practical throughput, numpy) and
is intentionally not run by the test suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

try:  # Works both as `python -m tools...` and `python tools/...`.
    from tools.turbo_lora_peft import (  # type: ignore
        LoraPair,
        ModuleSpec,
        TensorInfo,
        TurboLoraError,
        _json_dump,
        _tensor_info,
        expected_h3_module_specs,
        git_revision,
        load_json,
        parse_lora_pairs,
        read_safetensors_header,
        sha256_file,
        validate_module_map,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised only by direct script invocation from tools/
    from turbo_lora_peft import (  # type: ignore
        LoraPair,
        ModuleSpec,
        TensorInfo,
        TurboLoraError,
        _json_dump,
        _tensor_info,
        expected_h3_module_specs,
        git_revision,
        load_json,
        parse_lora_pairs,
        read_safetensors_header,
        sha256_file,
        validate_module_map,
    )

DEFAULT_BASE_FL2VA = Path("models/MiniMax-H3/FL2VA")
DEFAULT_LORA = Path("models/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors")
DEFAULT_OUTPUT_FL2VA = Path("models/MiniMax-H3-Turbo-Merged/FL2VA")
DEFAULT_MANIFEST = "merge_manifest.json"
DEFAULT_STRENGTH = 1.0
EXPECTED_TURBO_PAIR_COUNT = 259
EXPECTED_TRANSFORMER_SHARD_COUNT = 13
COMPONENT_LINKS = ("processor", "tokenizer", "text_encoder", "video_vae", "audio_vae")

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class TensorPlan:
    weight: str
    module: str
    shard: str
    spec: ModuleSpec
    pair: LoraPair
    lora_a_key: str
    lora_b_key: str


@dataclass(frozen=True)
class MergePlan:
    base_root: Path
    base_transformer: Path
    output_root: Path
    output_transformer: Path
    lora_path: Path
    config: dict[str, Any]
    index: dict[str, Any]
    specs: dict[str, ModuleSpec]
    pairs: dict[str, LoraPair]
    lora_header: dict[str, Any]
    lora_header_len: int
    lora_file_size: int
    validation: dict[str, Any]
    shard_names: list[str]
    targets_by_shard: dict[str, dict[str, TensorPlan]]
    strip_prefix: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _product(shape: Iterable[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise TurboLoraError(f"unsupported safetensors dtype {dtype!r}") from exc


def _expected_nbytes(info: TensorInfo) -> int:
    return _dtype_nbytes(info.dtype) * _product(info.shape)


def _require_offsets(path: Path, name: str, info: TensorInfo, *, file_size: int, header_len: int) -> tuple[int, int]:
    if info.data_offsets is None:
        raise TurboLoraError(f"{path}:{name} missing data_offsets")
    start, end = info.data_offsets
    expected = _expected_nbytes(info)
    if start < 0 or end < start:
        raise TurboLoraError(f"{path}:{name} invalid data_offsets {info.data_offsets}")
    if end - start != expected:
        raise TurboLoraError(f"{path}:{name} byte size mismatch: offsets={end - start}, dtype/shape={expected}")
    data_start = 8 + header_len
    if data_start + end > file_size:
        raise TurboLoraError(f"{path}:{name} data_offsets exceed file size")
    return start, end


def _lora_tensor_key(module: str, suffix: str, strip_prefix: str) -> str:
    prefix = strip_prefix.rstrip(".")
    return f"{prefix}.{module}.{suffix}" if prefix else f"{module}.{suffix}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_tensor_bytes(path: Path, header_len: int, info: TensorInfo) -> bytes:
    if info.data_offsets is None:
        raise TurboLoraError(f"{path} tensor missing data_offsets")
    start, end = info.data_offsets
    with path.open("rb") as f:
        f.seek(8 + header_len + start)
        data = f.read(end - start)
    if len(data) != end - start:
        raise TurboLoraError(f"{path} ended while reading tensor bytes")
    return data


def _copy_range_and_hash(src: BinaryIO, dst: BinaryIO, nbytes: int, *, chunk_size: int = 1024 * 1024) -> tuple[str, str]:
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    remaining = nbytes
    while remaining:
        chunk = src.read(min(chunk_size, remaining))
        if not chunk:
            raise TurboLoraError("source shard ended during tensor copy")
        dst.write(chunk)
        source_hash.update(chunk)
        output_hash.update(chunk)
        remaining -= len(chunk)
    return source_hash.hexdigest(), output_hash.hexdigest()


def _bf16_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(value) << 16))[0]


def _float_to_bf16(value: float) -> int:
    # Round float32 bits to nearest-even BF16.  This mirrors the common
    # float32->bfloat16 conversion used by tensor runtimes for finite values.
    f32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if (f32 & 0x7F800000) == 0x7F800000:  # inf/nan: preserve top payload bits
        return (f32 >> 16) & 0xFFFF
    lsb = (f32 >> 16) & 1
    return ((f32 + 0x7FFF + lsb) >> 16) & 0xFFFF


def _unpack_matrix(raw: bytes, dtype: str, shape: tuple[int, ...]) -> list[list[float]]:
    if len(shape) != 2:
        raise TurboLoraError(f"merge tensors must be 2D, got shape={shape}")
    rows, cols = shape
    if dtype == "BF16":
        vals = [_bf16_to_float(v) for (v,) in struct.iter_unpack("<H", raw)]
    elif dtype == "F32":
        vals = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    elif dtype == "F16":
        vals = [float(v) for (v,) in struct.iter_unpack("<e", raw)]
    else:
        raise TurboLoraError(f"pure-python merge supports BF16/F16/F32 tensors, got {dtype}")
    if len(vals) != rows * cols:
        raise TurboLoraError(f"raw tensor byte count does not match shape {shape}")
    return [vals[i * cols : (i + 1) * cols] for i in range(rows)]


def _pack_matrix(matrix: list[list[float]], dtype: str) -> bytes:
    flat = [v for row in matrix for v in row]
    if dtype == "BF16":
        return b"".join(struct.pack("<H", _float_to_bf16(v)) for v in flat)
    if dtype == "F32":
        return struct.pack(f"<{len(flat)}f", *flat)
    if dtype == "F16":
        return b"".join(struct.pack("<e", float(v)) for v in flat)
    raise TurboLoraError(f"pure-python merge supports BF16/F16/F32 tensors, got {dtype}")


def _merge_tensor_python(
    *,
    base_raw: bytes,
    a_raw: bytes,
    b_raw: bytes,
    base_info: TensorInfo,
    pair: LoraPair,
    strength: float,
) -> bytes:
    base = _unpack_matrix(base_raw, base_info.dtype, base_info.shape)
    a = _unpack_matrix(a_raw, pair.a.dtype, pair.a.shape)
    b = _unpack_matrix(b_raw, pair.b.dtype, pair.b.shape)
    out_features, in_features = base_info.shape
    rank = pair.rank
    result: list[list[float]] = []
    for o in range(out_features):
        row: list[float] = []
        brow = b[o]
        for i in range(in_features):
            delta = 0.0
            for r in range(rank):
                delta += brow[r] * a[r][i]
            row.append(base[o][i] + strength * delta)
        result.append(row)
    return _pack_matrix(result, base_info.dtype)


def _torch_tensor_from_raw(torch: Any, raw: bytes, dtype: str, shape: tuple[int, ...], device: str) -> Any:
    buf = bytearray(raw)  # torch.frombuffer warns on immutable bytes.
    if dtype == "BF16":
        tensor = torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16)
    elif dtype == "F16":
        tensor = torch.frombuffer(buf, dtype=torch.float16)
    elif dtype == "F32":
        tensor = torch.frombuffer(buf, dtype=torch.float32)
    else:
        raise TurboLoraError(f"torch merge supports BF16/F16/F32 target tensors, got {dtype}")
    return tensor.reshape(tuple(shape)).to(device)


def _torch_tensor_to_bytes(torch: Any, tensor: Any, dtype: str) -> bytes:
    cpu = tensor.detach().to("cpu").contiguous()
    if dtype == "BF16":
        cpu = cpu.to(torch.bfloat16)
    elif dtype == "F16":
        cpu = cpu.to(torch.float16)
    elif dtype == "F32":
        cpu = cpu.to(torch.float32)
    else:
        raise TurboLoraError(f"torch merge supports BF16/F16/F32 target tensors, got {dtype}")
    as_u8 = cpu.view(torch.uint8).reshape(-1)
    try:  # Fast path when numpy is available in the execution environment.
        return as_u8.numpy().tobytes()
    except Exception:  # pragma: no cover - slow fallback for unusual torch installs.
        return bytes(as_u8.tolist())


def _merge_tensor_torch(
    *,
    base_raw: bytes,
    a_raw: bytes,
    b_raw: bytes,
    base_info: TensorInfo,
    pair: LoraPair,
    strength: float,
    device: str,
) -> bytes:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - torch absent in CI fixture env.
        raise TurboLoraError("--engine torch requires torch; use --engine python only for tiny fixtures") from exc
    with torch.no_grad():
        base = _torch_tensor_from_raw(torch, base_raw, base_info.dtype, base_info.shape, device).float()
        a = _torch_tensor_from_raw(torch, a_raw, pair.a.dtype, pair.a.shape, device).float()
        b = _torch_tensor_from_raw(torch, b_raw, pair.b.dtype, pair.b.shape, device).float()
        merged = base.add(torch.matmul(b, a), alpha=float(strength))
        return _torch_tensor_to_bytes(torch, merged, base_info.dtype)


def _assert_single_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if device not in {"cuda", "cuda:0"}:
        raise TurboLoraError("single-card merge expects --device cuda:0 after CUDA_VISIBLE_DEVICES remapping")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
        if len(tokens) != 1:
            raise TurboLoraError(
                "single-card CUDA merge requires exactly one CUDA_VISIBLE_DEVICES entry; "
                f"got {visible!r}"
            )
        return
    try:
        import torch  # type: ignore

        count = torch.cuda.device_count()
    except Exception:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            count = len([line for line in output.splitlines() if line.strip()])
        except Exception as exc:
            raise TurboLoraError(
                "could not verify single visible CUDA device; set CUDA_VISIBLE_DEVICES to one GPU before --device cuda:0"
            ) from exc
    if count != 1:
        raise TurboLoraError(
            "single-card CUDA merge requires exactly one visible GPU; set CUDA_VISIBLE_DEVICES=<gpu> "
            f"before running (visible count={count})"
        )


def _normalize_resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _ensure_output_not_base(base_root: Path, base_transformer: Path, output_root: Path, output_transformer: Path) -> None:
    base_model = _normalize_resolved(base_root)
    out_model = _normalize_resolved(output_root)
    if out_model == base_model:
        raise TurboLoraError("refusing to write merged checkpoint over the official base model tree")
    for parent in (out_model, *out_model.parents):
        if parent == base_model:
            raise TurboLoraError("refusing to write merged checkpoint inside the official base model tree")

    base = _normalize_resolved(base_transformer)
    out = _normalize_resolved(output_transformer)
    if out == base:
        raise TurboLoraError("refusing to write merged transformer over the official base transformer")
    for parent in (out, *out.parents):
        if parent == base:
            raise TurboLoraError("refusing to write merged transformer inside the official base transformer tree")


def _validate_lora_offsets(lora_path: Path, header: dict[str, Any], header_len: int, file_size: int, pairs: dict[str, LoraPair], strip_prefix: str) -> None:
    for module, pair in pairs.items():
        for suffix, info in (("lora_A.weight", pair.a), ("lora_B.weight", pair.b)):
            key = _lora_tensor_key(module, suffix, strip_prefix)
            if key not in header:
                raise TurboLoraError(f"LoRA header missing tensor {key}")
            _require_offsets(lora_path, key, info, file_size=file_size, header_len=header_len)


def build_merge_plan(
    *,
    base_root: Path,
    lora_path: Path,
    output_root: Path,
    strip_prefix: str = "",
    expected_pair_count: int | None = EXPECTED_TURBO_PAIR_COUNT,
    expected_shard_count: int | None = EXPECTED_TRANSFORMER_SHARD_COUNT,
) -> MergePlan:
    base_transformer = base_root / "transformer"
    output_transformer = output_root / "transformer"
    _ensure_output_not_base(base_root, base_transformer, output_root, output_transformer)
    config_path = base_transformer / "config.json"
    index_path = base_transformer / "model.safetensors.index.json"
    config = load_json(config_path)
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TurboLoraError(f"{index_path} missing object weight_map")

    lora_header, lora_header_len, lora_file_size = read_safetensors_header(lora_path)
    pairs = parse_lora_pairs(lora_header, strip_prefix=strip_prefix)
    specs = expected_h3_module_specs(config)
    validation = validate_module_map(pairs, specs, index=index)
    if expected_pair_count is not None and validation["pair_count"] != expected_pair_count:
        raise TurboLoraError(f"expected {expected_pair_count} LoRA pairs, got {validation['pair_count']}")
    _validate_lora_offsets(lora_path, lora_header, lora_header_len, lora_file_size, pairs, strip_prefix)

    shard_names = sorted({shard for shard in weight_map.values() if isinstance(shard, str)})
    if expected_shard_count is not None and len(shard_names) != expected_shard_count:
        raise TurboLoraError(f"expected {expected_shard_count} transformer shards, got {len(shard_names)}")

    targets_by_shard: dict[str, dict[str, TensorPlan]] = {name: {} for name in shard_names}
    missing_base: list[str] = []
    for module, spec in specs.items():
        shard = weight_map.get(spec.base_weight)
        if not isinstance(shard, str):
            missing_base.append(spec.base_weight)
            continue
        targets_by_shard.setdefault(shard, {})[spec.base_weight] = TensorPlan(
            weight=spec.base_weight,
            module=module,
            shard=shard,
            spec=spec,
            pair=pairs[module],
            lora_a_key=_lora_tensor_key(module, "lora_A.weight", strip_prefix),
            lora_b_key=_lora_tensor_key(module, "lora_B.weight", strip_prefix),
        )
    if missing_base:
        raise TurboLoraError(f"base checkpoint index missing target weights: {missing_base[:8]}")

    base_header_shapes_checked = 0
    base_shards_read: list[str] = []
    for shard in shard_names:
        shard_path = base_transformer / shard
        header, header_len, file_size = read_safetensors_header(shard_path)
        base_shards_read.append(shard)
        for weight, target in targets_by_shard.get(shard, {}).items():
            entry = header.get(weight)
            if entry is None:
                raise TurboLoraError(f"{shard} missing target base weight {weight}")
            info = _tensor_info(weight, entry)
            _require_offsets(shard_path, weight, info, file_size=file_size, header_len=header_len)
            expected_shape = (target.spec.out_features, target.spec.in_features)
            if info.shape != expected_shape:
                raise TurboLoraError(f"{weight} shape mismatch in {shard}: expected {expected_shape}, got {info.shape}")
            if info.dtype != target.pair.a.dtype:
                raise TurboLoraError(f"{weight} dtype {info.dtype} does not match LoRA dtype {target.pair.a.dtype}")
            base_header_shapes_checked += 1
    validation.update(
        {
            "base_checkpoint_header_shapes_checked": base_header_shapes_checked,
            "base_checkpoint_shards_read": base_shards_read,
            "transformer_shard_count": len(shard_names),
            "target_weight_count": sum(len(v) for v in targets_by_shard.values()),
            "lora_safetensors_header_metadata": lora_header.get("__metadata__", {}),
            "lora_safetensors_header_len": lora_header_len,
            "lora_safetensors_size_bytes": lora_file_size,
            "strip_prefix": strip_prefix,
        }
    )

    return MergePlan(
        base_root=base_root,
        base_transformer=base_transformer,
        output_root=output_root,
        output_transformer=output_transformer,
        lora_path=lora_path,
        config=config,
        index=index,
        specs=specs,
        pairs=pairs,
        lora_header=lora_header,
        lora_header_len=lora_header_len,
        lora_file_size=lora_file_size,
        validation=validation,
        shard_names=shard_names,
        targets_by_shard=targets_by_shard,
        strip_prefix=strip_prefix,
    )


def _output_header_for_shard(input_header: dict[str, Any], tensor_names: list[str]) -> tuple[bytes, dict[str, Any]]:
    out_header: dict[str, Any] = {}
    metadata = dict(input_header.get("__metadata__", {})) if isinstance(input_header.get("__metadata__"), dict) else {}
    metadata.update(
        {
            "merged_from": "MiniMax-H3 FL2VA transformer + Turbo LoRA",
            "merge_formula": "W_eff = W + strength * (lora_B @ lora_A)",
            "merge_strength": str(DEFAULT_STRENGTH),
        }
    )
    out_header["__metadata__"] = metadata
    cursor = 0
    for name in tensor_names:
        info = _tensor_info(name, input_header[name])
        nbytes = _expected_nbytes(info)
        out_header[name] = {"dtype": info.dtype, "shape": list(info.shape), "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    raw = json.dumps(out_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw, out_header


def _merge_target_bytes(
    *,
    plan: MergePlan,
    target: TensorPlan,
    base_raw: bytes,
    base_info: TensorInfo,
    strength: float,
    engine: str,
    device: str,
) -> bytes:
    a_info = _tensor_info(target.lora_a_key, plan.lora_header[target.lora_a_key])
    b_info = _tensor_info(target.lora_b_key, plan.lora_header[target.lora_b_key])
    a_raw = _read_tensor_bytes(plan.lora_path, plan.lora_header_len, a_info)
    b_raw = _read_tensor_bytes(plan.lora_path, plan.lora_header_len, b_info)
    if engine == "python":
        return _merge_tensor_python(base_raw=base_raw, a_raw=a_raw, b_raw=b_raw, base_info=base_info, pair=target.pair, strength=strength)
    if engine == "torch":
        return _merge_tensor_torch(
            base_raw=base_raw,
            a_raw=a_raw,
            b_raw=b_raw,
            base_info=base_info,
            pair=target.pair,
            strength=strength,
            device=device,
        )
    raise TurboLoraError(f"unsupported merge engine {engine!r}")


def _write_shard_atomic(
    *,
    plan: MergePlan,
    shard: str,
    strength: float,
    engine: str,
    device: str,
    force: bool,
) -> dict[str, Any]:
    input_path = plan.base_transformer / shard
    output_path = plan.output_transformer / shard
    if output_path.exists() and not force:
        raise TurboLoraError(f"output shard exists and --force was not supplied: {output_path}")
    input_header, input_header_len, input_size = read_safetensors_header(input_path)
    tensor_names = [name for name in input_header.keys() if name != "__metadata__"]
    output_prefix, output_header = _output_header_for_shard(input_header, tensor_names)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    targets = plan.targets_by_shard.get(shard, {})
    tensor_rows: list[dict[str, Any]] = []

    plan.output_transformer.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("rb") as src, tmp_path.open("wb") as dst:
            dst.write(output_prefix)
            for name in tensor_names:
                entry = input_header[name]
                info = _tensor_info(name, entry)
                start, end = _require_offsets(input_path, name, info, file_size=input_size, header_len=input_header_len)
                nbytes = end - start
                src.seek(8 + input_header_len + start)
                if name in targets:
                    base_raw = src.read(nbytes)
                    if len(base_raw) != nbytes:
                        raise TurboLoraError(f"{input_path}:{name} ended during target tensor read")
                    merged = _merge_target_bytes(
                        plan=plan,
                        target=targets[name],
                        base_raw=base_raw,
                        base_info=info,
                        strength=strength,
                        engine=engine,
                        device=device,
                    )
                    if len(merged) != nbytes:
                        raise TurboLoraError(f"merged tensor {name} byte size changed from {nbytes} to {len(merged)}")
                    dst.write(merged)
                    tensor_rows.append(
                        {
                            "name": name,
                            "target": True,
                            "module": targets[name].module,
                            "rank": targets[name].pair.rank,
                            "shape": list(info.shape),
                            "dtype": info.dtype,
                            "bytes": nbytes,
                            "source_sha256": _sha256_bytes(base_raw),
                            "output_sha256": _sha256_bytes(merged),
                            "lora_A": targets[name].lora_a_key,
                            "lora_B": targets[name].lora_b_key,
                        }
                    )
                else:
                    source_hash, output_hash = _copy_range_and_hash(src, dst, nbytes)
                    tensor_rows.append(
                        {
                            "name": name,
                            "target": False,
                            "shape": list(info.shape),
                            "dtype": info.dtype,
                            "bytes": nbytes,
                            "source_sha256": source_hash,
                            "output_sha256": output_hash,
                            "copied_bitwise": source_hash == output_hash,
                        }
                    )
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "shard": shard,
        "source_path": str(input_path),
        "output_path": str(output_path),
        "source_size_bytes": input_size,
        "output_size_bytes": output_path.stat().st_size,
        "source_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "target_tensor_count": sum(1 for row in tensor_rows if row["target"]),
        "tensor_count": len(tensor_rows),
        "tensors": tensor_rows,
    }


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(_json_dump(data), encoding="utf-8")
    os.replace(tmp, path)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _manifest_base(plan: MergePlan, *, strength: float, engine: str, device: str, link_mode: str) -> dict[str, Any]:
    return {
        "schema_version": "minimax-h3-turbo-offline-merge-v1",
        "status": "in_progress",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "base": {"fl2va_root": str(plan.base_root), "transformer": str(plan.base_transformer)},
        "lora": {"path": str(plan.lora_path), "size_bytes": plan.lora_file_size, "sha256": None},
        "output": {"fl2va_root": str(plan.output_root), "transformer": str(plan.output_transformer)},
        "merge": {
            "formula": "W_eff = cast_to_base_dtype(float32(W) + strength * float32(lora_B @ lora_A))",
            "strength": strength,
            "engine": engine,
            "device": device,
            "accumulation": "torch engine uses FP32 matmul and FP32 base add before cast; python engine is for tiny fixtures only",
            "dynamic_bf16_lora_note": "Static FP32-accumulate merge can differ slightly from runtime dynamic BF16 LoRA application; this is a practical replacement for vLLM 0.26 mixed-rank PEFT, not an exact dynamic-kernel fidelity claim.",
        },
        "validation": plan.validation,
        "component_link_mode": link_mode,
        "completed_shards": {},
        "skipped_shards": [],
        "source_revision": {"worktree_git": git_revision()},
    }


def _same_manifest_contract(manifest: dict[str, Any], plan: MergePlan, strength: float, engine: str, device: str) -> bool:
    return (
        manifest.get("schema_version") == "minimax-h3-turbo-offline-merge-v1"
        and manifest.get("base", {}).get("transformer") == str(plan.base_transformer)
        and manifest.get("lora", {}).get("path") == str(plan.lora_path)
        and manifest.get("output", {}).get("transformer") == str(plan.output_transformer)
        and math.isclose(float(manifest.get("merge", {}).get("strength", -1)), float(strength))
        and manifest.get("merge", {}).get("engine") == engine
        and manifest.get("merge", {}).get("device") == device
    )


def _copy_transformer_metadata(plan: MergePlan) -> None:
    plan.output_transformer.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan.base_transformer / "config.json", plan.output_transformer / "config.json")
    shutil.copy2(plan.base_transformer / "model.safetensors.index.json", plan.output_transformer / "model.safetensors.index.json")


def _link_or_hardlink_components(base_root: Path, output_root: Path, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for name in COMPONENT_LINKS:
        src = base_root / name
        if not src.exists():
            continue
        dst = output_root / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                rows.append({"name": name, "mode": "existing_symlink", "path": str(dst), "target": str(src.resolve())})
                continue
            raise TurboLoraError(f"component destination already exists and is not the expected symlink: {dst}")
        if mode == "symlink":
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
            rows.append({"name": name, "mode": "symlink", "path": str(dst), "target": str(src.resolve())})
        elif mode == "hardlink":
            if src.is_file():
                os.link(src, dst)
                rows.append({"name": name, "mode": "hardlink_file", "path": str(dst), "target": str(src)})
            else:
                for item in src.rglob("*"):
                    rel = item.relative_to(src)
                    out_item = dst / rel
                    if item.is_dir():
                        out_item.mkdir(parents=True, exist_ok=True)
                    elif item.is_file():
                        out_item.parent.mkdir(parents=True, exist_ok=True)
                        os.link(item, out_item)
                rows.append({"name": name, "mode": "hardlink_tree", "path": str(dst), "target": str(src)})
        else:
            raise TurboLoraError(f"unsupported component link mode {mode!r}")
    return rows


def _write_root_metadata(plan: MergePlan, manifest: dict[str, Any]) -> None:
    output_root = plan.output_root
    base_index = load_json(plan.base_root / "model_index.json") if (plan.base_root / "model_index.json").exists() else {}
    model_index = dict(base_index)
    model_index["_merged_turbo_lora"] = {
        "schema_version": "minimax-h3-turbo-merged-model-index-v1",
        "base_fl2va_root": str(plan.base_root),
        "lora_path": str(plan.lora_path),
        "transformer": "independent merged safetensors shards",
        "merge_manifest": DEFAULT_MANIFEST,
        "formula": manifest["merge"]["formula"],
        "strength": manifest["merge"]["strength"],
        "note": manifest["merge"]["dynamic_bf16_lora_note"],
    }
    (output_root / "model_index.json").write_text(_json_dump(model_index), encoding="utf-8")
    (output_root / "merge_config.json").write_text(
        _json_dump(
            {
                "schema_version": "minimax-h3-turbo-merged-config-v1",
                "base_model": str(plan.base_root),
                "lora": str(plan.lora_path),
                "transformer_config": "transformer/config.json",
                "transformer_index": "transformer/model.safetensors.index.json",
                "merge_manifest": DEFAULT_MANIFEST,
            }
        ),
        encoding="utf-8",
    )
    (output_root / "PROVENANCE.json").write_text(_json_dump(manifest), encoding="utf-8")
    (output_root / "LICENSE.upstream.md").write_text(
        "# Upstream license and provenance\n\n"
        "This directory is generated from the locally authorized MiniMax-H3 FL2VA base checkpoint and "
        "the local MiniMax-H3 Turbo LoRA checkpoint. It does not grant redistribution rights. Preserve "
        "and follow the upstream MiniMax-H3 / Turbo-LoRA license, access, and territory terms.\n\n"
        "The transformer shards are independently merged outputs. Non-transformer components are "
        "read-only links or hardlinks to the official local base tree as recorded in PROVENANCE.json.\n",
        encoding="utf-8",
    )


def merge_checkpoint(
    *,
    base_root: Path = DEFAULT_BASE_FL2VA,
    lora_path: Path = DEFAULT_LORA,
    output_root: Path = DEFAULT_OUTPUT_FL2VA,
    manifest_path: Path | None = None,
    strength: float = DEFAULT_STRENGTH,
    engine: str = "python",
    device: str = "cpu",
    strip_prefix: str = "",
    link_mode: str = "symlink",
    resume: bool = False,
    force: bool = False,
    expected_pair_count: int | None = EXPECTED_TURBO_PAIR_COUNT,
    expected_shard_count: int | None = EXPECTED_TRANSFORMER_SHARD_COUNT,
    max_shards: int | None = None,
    compute_lora_sha256: bool = False,
) -> dict[str, Any]:
    if strength != DEFAULT_STRENGTH:
        raise TurboLoraError("this mission contract requires strength=1.0")
    _assert_single_cuda_device(device)
    plan = build_merge_plan(
        base_root=base_root,
        lora_path=lora_path,
        output_root=output_root,
        strip_prefix=strip_prefix,
        expected_pair_count=expected_pair_count,
        expected_shard_count=expected_shard_count,
    )
    manifest_file = manifest_path or (plan.output_root / DEFAULT_MANIFEST)
    existing = _load_manifest(manifest_file) if resume else None
    if existing is not None:
        if not _same_manifest_contract(existing, plan, strength, engine, device):
            raise TurboLoraError(f"resume manifest contract does not match requested merge: {manifest_file}")
        manifest = existing
        manifest.setdefault("completed_shards", {})
        manifest.setdefault("skipped_shards", [])
        manifest["status"] = "in_progress"
        manifest["updated_at_utc"] = _utc_now()
    else:
        manifest = _manifest_base(plan, strength=strength, engine=engine, device=device, link_mode=link_mode)
    if compute_lora_sha256:
        manifest["lora"]["sha256"] = sha256_file(plan.lora_path)

    _copy_transformer_metadata(plan)
    processed = 0
    for shard in plan.shard_names:
        completed = manifest.get("completed_shards", {})
        output_path = plan.output_transformer / shard
        if resume and shard in completed and output_path.exists():
            recorded_hash = completed[shard].get("output_sha256")
            if recorded_hash and sha256_file(output_path) == recorded_hash:
                manifest.setdefault("skipped_shards", []).append(shard)
                continue
        shard_row = _write_shard_atomic(plan=plan, shard=shard, strength=strength, engine=engine, device=device, force=force or resume)
        manifest.setdefault("completed_shards", {})[shard] = shard_row
        manifest["updated_at_utc"] = _utc_now()
        _write_json_atomic(manifest_file, manifest)
        processed += 1
        if max_shards is not None and processed >= max_shards:
            manifest["status"] = "partial"
            manifest["updated_at_utc"] = _utc_now()
            _write_json_atomic(manifest_file, manifest)
            return manifest

    linked = _link_or_hardlink_components(plan.base_root, plan.output_root, link_mode)
    manifest["component_links"] = linked
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = _utc_now()
    manifest["updated_at_utc"] = manifest["completed_at_utc"]
    _write_json_atomic(manifest_file, manifest)
    _write_root_metadata(plan, manifest)
    return manifest


def validate_only(
    *,
    base_root: Path = DEFAULT_BASE_FL2VA,
    lora_path: Path = DEFAULT_LORA,
    output_root: Path = DEFAULT_OUTPUT_FL2VA,
    strip_prefix: str = "",
    expected_pair_count: int | None = EXPECTED_TURBO_PAIR_COUNT,
    expected_shard_count: int | None = EXPECTED_TRANSFORMER_SHARD_COUNT,
) -> dict[str, Any]:
    plan = build_merge_plan(
        base_root=base_root,
        lora_path=lora_path,
        output_root=output_root,
        strip_prefix=strip_prefix,
        expected_pair_count=expected_pair_count,
        expected_shard_count=expected_shard_count,
    )
    return {
        "schema_version": "minimax-h3-turbo-offline-merge-validate-v1",
        "status": "validated_static_headers_only",
        "base": {"fl2va_root": str(plan.base_root), "transformer": str(plan.base_transformer)},
        "lora": {"path": str(plan.lora_path), "size_bytes": plan.lora_file_size},
        "output": {"fl2va_root": str(plan.output_root), "transformer": str(plan.output_transformer)},
        "validation": plan.validation,
        "merge_contract": {
            "formula": "W_eff = cast_to_base_dtype(float32(W) + strength * float32(lora_B @ lora_A))",
            "strength": DEFAULT_STRENGTH,
            "safe_write": "per-shard temporary file followed by atomic rename; official base tree is never a destination",
            "resume": "merge_manifest.json records completed shards and per-tensor sha256/bytes",
        },
    }


def _optional_count(value: str) -> int | None:
    if value.lower() in {"none", "off", "disable", "disabled"}:
        return None
    return int(value)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-root", type=Path, default=DEFAULT_BASE_FL2VA)
        p.add_argument("--lora", type=Path, default=DEFAULT_LORA)
        p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_FL2VA)
        p.add_argument("--strip-prefix", default="")
        p.add_argument("--expected-pair-count", type=_optional_count, default=EXPECTED_TURBO_PAIR_COUNT)
        p.add_argument("--expected-shard-count", type=_optional_count, default=EXPECTED_TRANSFORMER_SHARD_COUNT)

    p_validate = sub.add_parser("validate", help="header/mapping validation only; no checkpoint writes")
    common(p_validate)

    p_merge = sub.add_parser("merge", help="create the independent merged FL2VA checkpoint")
    common(p_merge)
    p_merge.add_argument("--manifest", type=Path)
    p_merge.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    p_merge.add_argument("--engine", choices=["python", "torch"], default="torch")
    p_merge.add_argument("--device", default="cpu", help="cpu or cuda:0; CUDA path requires exactly one visible GPU")
    p_merge.add_argument("--link-mode", choices=["symlink", "hardlink"], default="symlink")
    p_merge.add_argument("--resume", action="store_true")
    p_merge.add_argument("--force", action="store_true")
    p_merge.add_argument("--sha256-lora", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.cmd == "validate":
            manifest = validate_only(
                base_root=args.base_root,
                lora_path=args.lora,
                output_root=args.output_root,
                strip_prefix=args.strip_prefix,
                expected_pair_count=args.expected_pair_count,
                expected_shard_count=args.expected_shard_count,
            )
        else:
            manifest = merge_checkpoint(
                base_root=args.base_root,
                lora_path=args.lora,
                output_root=args.output_root,
                manifest_path=args.manifest,
                strength=args.strength,
                engine=args.engine,
                device=args.device,
                strip_prefix=args.strip_prefix,
                link_mode=args.link_mode,
                resume=args.resume,
                force=args.force,
                expected_pair_count=args.expected_pair_count,
                expected_shard_count=args.expected_shard_count,
                compute_lora_sha256=args.sha256_lora,
            )
    except TurboLoraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_json_dump(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
