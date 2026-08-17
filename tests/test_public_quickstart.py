from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts" / "build_runtime.sh",
    ROOT / "scripts" / "prepare_models.sh",
    ROOT / "scripts" / "run_turbo_demo.sh",
)
EXAMPLE = ROOT / "examples" / "a6000-turbo-8step-sci-fi"
FL2VA_EXAMPLE = ROOT / "examples" / "a6000-turbo-8step-niulai-inspired"


def test_public_scripts_are_valid_and_dry_run_without_side_effects() -> None:
    subprocess.run(["bash", "-n", *map(str, SCRIPTS)], check=True)
    for script in SCRIPTS:
        proc = subprocess.run(
            ["bash", str(script), "--dry-run"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        assert "DRY-RUN" in proc.stdout


def test_public_example_metadata_matches_files() -> None:
    metadata = json.loads((EXAMPLE / "metadata.json").read_text(encoding="utf-8"))
    video = EXAMPLE / metadata["files"]["video"]["path"]
    contact = EXAMPLE / metadata["files"]["contact_sheet"]["path"]
    prompt = EXAMPLE / metadata["files"]["prompt"]["path"]

    assert video.stat().st_size == metadata["files"]["video"]["bytes"]
    assert hashlib.sha256(video.read_bytes()).hexdigest() == metadata["files"]["video"]["sha256"]
    assert hashlib.sha256(contact.read_bytes()).hexdigest() == metadata["files"]["contact_sheet"]["sha256"]
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == metadata["files"]["prompt"]["sha256"]
    assert metadata["validation"]["structural_av_contract_pass"] is True
    assert metadata["workload"]["decoded_video_frames"] == 124
    assert metadata["workload"]["audio_channels"] == 2


def test_public_fl2va_showcase_metadata_matches_files() -> None:
    metadata = json.loads((FL2VA_EXAMPLE / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["workload"]["task"] == "fl2va"
    assert metadata["workload"]["decoded_video_frames"] == 124
    assert metadata["workload"]["audio_channels"] == 2
    assert metadata["reference"]["included_in_public_release"] is False
    assert metadata["selection"]["candidate_seeds"] == [17, 42, 137]
    assert metadata["selection"]["selected_seed"] == 42
    assert metadata["validation"]["structural_av_contract_pass"] is True
    for record in metadata["files"].values():
        path = FL2VA_EXAMPLE / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_public_architecture_svg_is_self_contained_and_readable() -> None:
    svg = ROOT / "docs" / "assets" / "minimax-h3-a6000-pipeline.svg"
    parsed = ET.parse(svg)
    root = parsed.getroot()
    text = svg.read_text(encoding="utf-8")
    assert root.attrib["viewBox"] == "0 0 1600 900"
    assert "MODEL TO VERIFIED AUDIOVISUAL OUTPUT" in text
    assert "1792.202" in text and "4.326" in text and "4.316" in text
    assert "foreignObject" not in text and "<script" not in text and "<image" not in text
    assert "href=" not in text and "https://" not in text


def test_readmes_use_a6000_evidence_and_link_the_example() -> None:
    zh = (ROOT / "README.md").read_text(encoding="utf-8")
    en = (ROOT / "README_EN.md").read_text(encoding="utf-8")
    for text in (zh, en):
        assert "orbital-shipyard-turbo-8step.mp4" in text
        assert "niulai-inspired-forest-awakening-turbo-8step.mp4" in text
        assert "docs/assets/minimax-h3-a6000-pipeline.svg" in text
        assert "305.386" in text
        assert "1792.202" in text
        assert "290.998" in text
        assert "6.159" in text
        assert "15.203" in text
        assert "4.326" in text
        assert "CURRENT_WORK.md" in text
        assert "auto" + "nomously by" not in text.lower()
        assert "47 minutes 58.7" not in text
        assert "M4 Pro" not in text


def test_generation_runner_keeps_single_gpu_and_weights_read_only() -> None:
    script = (ROOT / "scripts" / "run_turbo_demo.sh").read_text(encoding="utf-8")
    assert '--gpus "device=$GPU_INDEX"' in script
    assert '--network none' in script
    assert '-v "$MODEL_DIR":/models/Turbo/FL2VA:ro' in script
    assert "torch.cuda.device_count() == 1" in script
    assert "INPUT_REFERENCE" in script
    assert '"task":"fl2va"' in script
    assert "input_reference=@/evidence/input_reference.png" in script
    assert "selected GPU" in script and "already has a compute process" in script


def test_repository_contains_full_apache_license() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
