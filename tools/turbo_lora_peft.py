#!/usr/bin/env python3
"""Static MiniMax-H3 Turbo LoRA PEFT converter/validator.

This module intentionally avoids torch, safetensors, GPU probing, model loads,
and network access.  It reads only safetensors headers plus JSON config/index
files, then either prepares a PEFT adapter directory or blocks with an explicit
runtime-compatibility reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SOURCE = Path("models/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors")
DEFAULT_H3_CONFIG = Path("models/MiniMax-H3/FL2VA/transformer/config.json")
DEFAULT_H3_INDEX = Path("models/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json")
DEFAULT_RUNTIME_SOURCE = Path("runtime/single_a6000_bf16/source_commit.json")
DEFAULT_OUTPUT_DIR = Path("models/MiniMax-H3-Turbo-Lora/peft")
DEFAULT_LOCK = Path("turbo_converted_lock.json")

ADAPTER_MODEL = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
GLOBAL_RANK = 64
LOW_RANK = 16

# Evidence from the pinned vLLM 0.26.0 source in the runtime image:
# - PEFTHelper dataclass fields are r/lora_alpha/target_modules plus feature
#   flags; rank_pattern/alpha_pattern are filtered out by from_dict().
# - LoRALayerWeights.from_config initializes every module with peft_helper.r.
PINNED_VLLM_SUPPORTS_RANK_PATTERN = False
PINNED_VLLM_RANK_PATTERN_BLOCK_REASON = (
    "pinned vLLM 0.26.0 PEFTHelper does not define rank_pattern/alpha_pattern "
    "and LoRALayerWeights.from_config uses the single global peft_helper.r for every module"
)


class TurboLoraError(ValueError):
    """Validation/conversion failed without touching model weights."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int] | None = None


@dataclass(frozen=True)
class LoraPair:
    module: str
    a: TensorInfo
    b: TensorInfo

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def in_features(self) -> int:
        return self.a.shape[1]

    @property
    def out_features(self) -> int:
        return self.b.shape[0]


@dataclass(frozen=True)
class ModuleSpec:
    module: str
    in_features: int
    out_features: int
    expected_rank: int
    base_weight: str


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path = Path(".")) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise TurboLoraError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise TurboLoraError(f"{path} ended before the safetensors header was complete")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise TurboLoraError(f"{path} has an invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise TurboLoraError(f"{path} safetensors header is not a JSON object")
    return header, header_len, path.stat().st_size


def _tensor_info(name: str, value: Any) -> TensorInfo:
    if not isinstance(value, dict):
        raise TurboLoraError(f"tensor {name} header entry is not an object")
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")
    if not isinstance(dtype, str):
        raise TurboLoraError(f"tensor {name} missing string dtype")
    if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
        raise TurboLoraError(f"tensor {name} missing integer shape list")
    parsed_offsets = None
    if offsets is not None:
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
            raise TurboLoraError(f"tensor {name} has invalid data_offsets")
        parsed_offsets = (offsets[0], offsets[1])
    return TensorInfo(dtype=dtype, shape=tuple(shape), data_offsets=parsed_offsets)


def parse_lora_pairs(header: dict[str, Any], *, strip_prefix: str = "") -> dict[str, LoraPair]:
    partial: dict[str, dict[str, TensorInfo]] = {}
    unexpected: list[str] = []
    prefix = strip_prefix.rstrip(".")
    for key, value in header.items():
        if key == "__metadata__":
            continue
        normalized = key
        if prefix:
            prefix_dot = prefix + "."
            if not normalized.startswith(prefix_dot):
                unexpected.append(key)
                continue
            normalized = normalized[len(prefix_dot) :]
        if normalized.endswith(".lora_A.weight"):
            module = normalized[: -len(".lora_A.weight")]
            partial.setdefault(module, {})["A"] = _tensor_info(key, value)
        elif normalized.endswith(".lora_B.weight"):
            module = normalized[: -len(".lora_B.weight")]
            partial.setdefault(module, {})["B"] = _tensor_info(key, value)
        else:
            unexpected.append(key)
    if unexpected:
        raise TurboLoraError(f"unexpected non-LoRA tensor keys: {unexpected[:8]}")

    missing = [module for module, sides in partial.items() if set(sides) != {"A", "B"}]
    if missing:
        raise TurboLoraError(f"incomplete LoRA A/B pairs for modules: {missing[:8]}")

    pairs: dict[str, LoraPair] = {}
    for module, sides in partial.items():
        a = sides["A"]
        b = sides["B"]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise TurboLoraError(f"{module} LoRA tensors must be 2D, got A={a.shape}, B={b.shape}")
        if a.shape[0] != b.shape[1]:
            raise TurboLoraError(f"{module} rank mismatch: A={a.shape}, B={b.shape}")
        if a.dtype != b.dtype:
            raise TurboLoraError(f"{module} dtype mismatch: A={a.dtype}, B={b.dtype}")
        pairs[module] = LoraPair(module=module, a=a, b=b)
    return pairs


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TurboLoraError(f"{path} is not a JSON object")
    return data


def expected_h3_module_specs(config: dict[str, Any]) -> dict[str, ModuleSpec]:
    hidden = int(config["hidden_size"])
    layers = int(config["num_layers"])
    refiner_layers = int(config["token_refiner_num_layers"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    ffn = int(config["ffn_hidden_size"])
    time_dim = int(config["time_embed_dim"])
    adaln_out = int(config["adaln_out_features"])
    final_adaln_out = int(config["final_adaln_out_features"])
    qkv_out = 3 * heads * head_dim
    attn_out_in = heads * head_dim
    specs: dict[str, ModuleSpec] = {}

    def add(module: str, in_features: int, out_features: int, rank: int) -> None:
        specs[module] = ModuleSpec(
            module=module,
            in_features=in_features,
            out_features=out_features,
            expected_rank=rank,
            base_weight=f"{module}.weight",
        )

    for i in range(layers):
        prefix = f"blocks.{i}"
        add(f"{prefix}.adaln_proj.linear", time_dim, adaln_out, LOW_RANK)
        add(f"{prefix}.attn.out_proj", attn_out_in, hidden, GLOBAL_RANK)
        add(f"{prefix}.attn.qkv_proj", hidden, qkv_out, GLOBAL_RANK)
        add(f"{prefix}.mlp.fc1", hidden, 2 * ffn, GLOBAL_RANK)
        add(f"{prefix}.mlp.fc2", ffn, hidden, GLOBAL_RANK)
    for i in range(refiner_layers):
        prefix = f"token_refiner.blocks.{i}"
        add(f"{prefix}.attn.out_proj", attn_out_in, hidden, GLOBAL_RANK)
        add(f"{prefix}.attn.qkv_proj", hidden, qkv_out, GLOBAL_RANK)
        add(f"{prefix}.mlp.fc1", hidden, 2 * ffn, GLOBAL_RANK)
        add(f"{prefix}.mlp.fc2", ffn, hidden, GLOBAL_RANK)
    add("final_layer.adaln_proj.linear", time_dim, final_adaln_out, LOW_RANK)
    return specs


def validate_module_map(
    pairs: dict[str, LoraPair],
    specs: dict[str, ModuleSpec],
    *,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_names = set(pairs)
    spec_names = set(specs)
    missing = sorted(spec_names - pair_names)
    unexpected = sorted(pair_names - spec_names)
    shape_mismatches: list[dict[str, Any]] = []
    dtype_mismatches: list[dict[str, Any]] = []
    rank_mismatches: list[dict[str, Any]] = []
    for module in sorted(pair_names & spec_names):
        pair = pairs[module]
        spec = specs[module]
        if pair.in_features != spec.in_features or pair.out_features != spec.out_features:
            shape_mismatches.append(
                {
                    "module": module,
                    "expected": [spec.out_features, spec.in_features],
                    "lora_A": list(pair.a.shape),
                    "lora_B": list(pair.b.shape),
                }
            )
        if pair.rank != spec.expected_rank:
            rank_mismatches.append({"module": module, "expected_rank": spec.expected_rank, "rank": pair.rank})
        if pair.a.dtype != "BF16" or pair.b.dtype != "BF16":
            dtype_mismatches.append({"module": module, "A_dtype": pair.a.dtype, "B_dtype": pair.b.dtype})

    missing_base_weights: list[str] = []
    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise TurboLoraError("base checkpoint index missing object weight_map")
        for spec in specs.values():
            if spec.base_weight not in weight_map:
                missing_base_weights.append(spec.base_weight)

    errors = {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "rank_mismatches": rank_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "missing_base_weights": missing_base_weights,
    }
    if any(errors.values()):
        raise TurboLoraError("module map validation failed: " + _json_dump(errors))

    rank_hist: dict[str, int] = {}
    target_suffixes: set[str] = set()
    for pair in pairs.values():
        rank_hist[str(pair.rank)] = rank_hist.get(str(pair.rank), 0) + 1
        target_suffixes.add(pair.module.rsplit(".", 1)[-1])

    return {
        "pair_count": len(pairs),
        "rank_histogram": dict(sorted(rank_hist.items(), key=lambda kv: int(kv[0]))),
        "target_suffixes": sorted(target_suffixes),
        "base_weight_count_checked": len(specs) if index is not None else 0,
    }


def validate_base_checkpoint_header_shapes(index_path: Path, specs: dict[str, ModuleSpec]) -> dict[str, Any]:
    """Verify target base weights exist in sharded safetensors headers and match shapes."""
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TurboLoraError(f"{index_path} missing object weight_map")

    wanted_by_shard: dict[str, list[ModuleSpec]] = {}
    missing_from_index: list[str] = []
    for spec in specs.values():
        shard = weight_map.get(spec.base_weight)
        if not isinstance(shard, str):
            missing_from_index.append(spec.base_weight)
            continue
        wanted_by_shard.setdefault(shard, []).append(spec)
    if missing_from_index:
        raise TurboLoraError(f"base checkpoint index missing target weights: {missing_from_index[:8]}")

    shape_mismatches: list[dict[str, Any]] = []
    missing_from_shard: list[str] = []
    shards_read: list[str] = []
    for shard, shard_specs in sorted(wanted_by_shard.items()):
        shard_path = index_path.parent / shard
        header, _, _ = read_safetensors_header(shard_path)
        shards_read.append(shard)
        for spec in shard_specs:
            entry = header.get(spec.base_weight)
            if entry is None:
                missing_from_shard.append(spec.base_weight)
                continue
            info = _tensor_info(spec.base_weight, entry)
            expected = (spec.out_features, spec.in_features)
            if info.shape != expected:
                shape_mismatches.append(
                    {
                        "weight": spec.base_weight,
                        "shard": shard,
                        "expected_shape": list(expected),
                        "actual_shape": list(info.shape),
                    }
                )
    if missing_from_shard or shape_mismatches:
        raise TurboLoraError(
            "base checkpoint header validation failed: "
            + _json_dump({"missing_from_shard": missing_from_shard, "shape_mismatches": shape_mismatches})
        )
    return {
        "base_checkpoint_header_shapes_checked": len(specs),
        "base_checkpoint_shards_read": shards_read,
    }


def build_adapter_config(pairs: dict[str, LoraPair], *, base_model_name: str = "MiniMax-H3") -> dict[str, Any]:
    # lora_alpha == rank keeps the safetensors metadata contract W + B @ A.
    rank_pattern = {module: pair.rank for module, pair in sorted(pairs.items()) if pair.rank != GLOBAL_RANK}
    alpha_pattern = dict(rank_pattern)
    return {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": GLOBAL_RANK,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": GLOBAL_RANK,
        "target_modules": ["fc1", "fc2", "linear", "out_proj", "qkv_proj"],
        "task_type": None,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }


def runtime_support_report() -> dict[str, Any]:
    return {
        "vllm_supports_rank_pattern": PINNED_VLLM_SUPPORTS_RANK_PATTERN,
        "mixed_rank_block_reason": PINNED_VLLM_RANK_PATTERN_BLOCK_REASON,
        "required_for_this_adapter": "mixed ranks 64 and 16 with rank_pattern/alpha_pattern",
        "pinned_source_evidence": [
            "runtime/single_a6000_bf16/src/vllm-omni/vllm_omni/diffusion/lora/manager.py:_load_adapter uses PEFTHelper.from_local_dir and LoRAModel.from_local_checkpoint",
            "runtime image vllm.lora.peft_helper.PEFTHelper fields omit rank_pattern/alpha_pattern",
            "runtime image vllm.lora.lora_weights.LoRALayerWeights.from_config uses peft_helper.r for every module",
        ],
    }


def build_manifest(
    *,
    source: Path,
    header_len: int,
    file_size: int,
    tensor_count: int,
    source_sha256: str | None,
    validation: dict[str, Any],
    adapter_config: dict[str, Any],
    runtime_source: dict[str, Any] | None,
    output_dir: Path | None,
    status: str,
    block_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "minimax-h3-turbo-peft-lock-v1",
        "status": status,
        "block_reason": block_reason,
        "source": {
            "path": str(source),
            "size_bytes": file_size,
            "sha256": source_sha256,
            "safetensors_header_len": header_len,
            "tensor_count": tensor_count,
        },
        "adapter": {
            "output_dir": str(output_dir) if output_dir is not None else None,
            "adapter_model": ADAPTER_MODEL,
            "adapter_config": adapter_config,
            "asset_mode": None,
        },
        "validation": validation,
        "runtime_support": runtime_support_report(),
        "source_revision": {
            "worktree_git": git_revision(),
            "runtime_source": runtime_source,
        },
        "license": {
            "converter": "project-local; no separate license header beyond repository policy",
            "vllm_omni": "Apache-2.0 per pyproject.toml",
            "model_or_lora": "preserve upstream MiniMax-H3 / Turbo-LoRA license and access terms; no redistribution performed by this tool",
        },
        "deployment_draft": {
            "script": "scripts/run_gpu2_turbo_lora_vllm_omni_draft.sh",
            "profile": "GPU index 2 draft; official FL2VA partition; Turbo-LoRA only; 4-step and 8-step requests; DLO enabled; dense CUDNN_ATTN; no SolAttn/cache path",
            "guard": "script refuses to run unless this lock status is converted_static_validation_passed",
        },
        "fidelity_comparison_plan": {
            "baseline": "existing 50-step FL2VA/T2VA seed-0 baseline artifacts under technical_report/evidence/minimax_h3_desktop/baseline_a6000",
            "turbo_variants": ["4 inference steps", "8 inference steps"],
            "controlled_inputs": ["same prompt", "same seed", "1344x768", "124 frames at 24 fps", "flow_shift=12", "audio_flow_shift=3.0", "quality=lossless", "task=t2va"],
            "static_acceptance_before_gpu": ["adapter header/module-map validation passes", "runtime mixed-rank support is resolved or conversion remains blocked"],
            "post_gpu_checks": ["HTTP metrics and resource monitor", "MP4 decode contract: H.264 video plus 32 kHz stereo audio", "output hashes", "side-by-side qualitative comparison against 50-step baseline", "document latency/quality tradeoff separately for 4-step and 8-step"],
        },
    }


def _prepare(source: Path, h3_config: Path, h3_index: Path | None, strip_prefix: str) -> tuple[dict[str, Any], int, int, dict[str, LoraPair], dict[str, Any]]:
    header, header_len, file_size = read_safetensors_header(source)
    tensor_count = len([k for k in header if k != "__metadata__"])
    pairs = parse_lora_pairs(header, strip_prefix=strip_prefix)
    config = load_json(h3_config)
    index = load_json(h3_index) if h3_index is not None else None
    specs = expected_h3_module_specs(config)
    validation = validate_module_map(pairs, specs, index=index)
    if h3_index is not None:
        validation.update(validate_base_checkpoint_header_shapes(h3_index, specs))
    validation.update(
        {
            "safetensors_header_metadata": header.get("__metadata__", {}),
            "safetensors_tensor_count": tensor_count,
            "strip_prefix": strip_prefix,
        }
    )
    return header, header_len, file_size, pairs, validation


def convert_adapter(
    *,
    source: Path,
    output_dir: Path,
    h3_config: Path,
    h3_index: Path | None,
    mode: str,
    force: bool,
    allow_unsupported_mixed_rank: bool,
    lock_out: Path | None,
    strip_prefix: str = "",
    compute_sha256: bool = True,
) -> dict[str, Any]:
    header, header_len, file_size, pairs, validation = _prepare(source, h3_config, h3_index, strip_prefix)
    adapter_config = build_adapter_config(pairs)
    runtime_source = load_json(DEFAULT_RUNTIME_SOURCE) if DEFAULT_RUNTIME_SOURCE.exists() else None
    tensor_count = validation["safetensors_tensor_count"]
    source_hash = sha256_file(source) if compute_sha256 else None

    mixed = len({pair.rank for pair in pairs.values()}) > 1
    block_reason = None
    if mixed and not PINNED_VLLM_SUPPORTS_RANK_PATTERN and not allow_unsupported_mixed_rank:
        block_reason = PINNED_VLLM_RANK_PATTERN_BLOCK_REASON
        manifest = build_manifest(
            source=source,
            header_len=header_len,
            file_size=file_size,
            tensor_count=tensor_count,
            source_sha256=source_hash,
            validation=validation,
            adapter_config=adapter_config,
            runtime_source=runtime_source,
            output_dir=output_dir,
            status="blocked_mixed_rank_runtime_unsupported",
            block_reason=block_reason,
        )
        if lock_out is not None:
            lock_out.write_text(_json_dump(manifest), encoding="utf-8")
        raise TurboLoraError(block_reason)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / ADAPTER_CONFIG
    model_path = output_dir / ADAPTER_MODEL
    if (config_path.exists() or model_path.exists()) and not force:
        raise TurboLoraError(f"{output_dir} already contains adapter files; pass --force to replace them")
    config_path.write_text(_json_dump(adapter_config), encoding="utf-8")
    if model_path.exists() or model_path.is_symlink():
        model_path.unlink()
    if mode == "symlink":
        model_path.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, model_path)
    else:
        raise TurboLoraError(f"unsupported asset mode {mode!r}")

    manifest = build_manifest(
        source=source,
        header_len=header_len,
        file_size=file_size,
        tensor_count=tensor_count,
        source_sha256=source_hash,
        validation=validation,
        adapter_config=adapter_config,
        runtime_source=runtime_source,
        output_dir=output_dir,
        status="converted_static_validation_passed",
        block_reason=None,
    )
    manifest["adapter"]["asset_mode"] = mode
    manifest["adapter"]["adapter_model_resolved"] = str(model_path.resolve())
    if lock_out is not None:
        lock_out.write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def validate_only(
    *,
    source: Path,
    h3_config: Path,
    h3_index: Path | None,
    lock_out: Path | None,
    strip_prefix: str = "",
    compute_sha256: bool = False,
) -> dict[str, Any]:
    header, header_len, file_size, pairs, validation = _prepare(source, h3_config, h3_index, strip_prefix)
    adapter_config = build_adapter_config(pairs)
    source_hash = sha256_file(source) if compute_sha256 else None
    mixed = len({pair.rank for pair in pairs.values()}) > 1
    status = "validated_static_only"
    block_reason = None
    if mixed and not PINNED_VLLM_SUPPORTS_RANK_PATTERN:
        status = "blocked_mixed_rank_runtime_unsupported"
        block_reason = PINNED_VLLM_RANK_PATTERN_BLOCK_REASON
    manifest = build_manifest(
        source=source,
        header_len=header_len,
        file_size=file_size,
        tensor_count=validation["safetensors_tensor_count"],
        source_sha256=source_hash,
        validation=validation,
        adapter_config=adapter_config,
        runtime_source=load_json(DEFAULT_RUNTIME_SOURCE) if DEFAULT_RUNTIME_SOURCE.exists() else None,
        output_dir=None,
        status=status,
        block_reason=block_reason,
    )
    if lock_out is not None:
        lock_out.write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--h3-config", type=Path, default=DEFAULT_H3_CONFIG)
    parser.add_argument("--h3-index", type=Path, default=DEFAULT_H3_INDEX)
    parser.add_argument("--strip-prefix", default="", help="explicit prefix to strip from every LoRA tensor key; default strips nothing")
    parser.add_argument("--lock-out", type=Path)
    parser.add_argument("--sha256", action="store_true", help="compute full source SHA256 (reads the complete local file)")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="static header/module-map validation only")
    _add_common_args(p_validate)

    p_convert = sub.add_parser("convert", help="create PEFT adapter directory when runtime-compatible")
    _add_common_args(p_convert)
    p_convert.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_convert.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p_convert.add_argument("--force", action="store_true")
    p_convert.add_argument(
        "--allow-unsupported-mixed-rank",
        action="store_true",
        help="dangerous: write files even though pinned vLLM ignores rank_pattern; not for deployment",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.cmd == "validate":
            manifest = validate_only(
                source=args.source,
                h3_config=args.h3_config,
                h3_index=args.h3_index,
                lock_out=args.lock_out,
                strip_prefix=args.strip_prefix,
                compute_sha256=args.sha256,
            )
        else:
            manifest = convert_adapter(
                source=args.source,
                output_dir=args.output_dir,
                h3_config=args.h3_config,
                h3_index=args.h3_index,
                mode=args.mode,
                force=args.force,
                allow_unsupported_mixed_rank=args.allow_unsupported_mixed_rank,
                lock_out=args.lock_out,
                strip_prefix=args.strip_prefix,
                compute_sha256=args.sha256,
            )
    except TurboLoraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_json_dump(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
