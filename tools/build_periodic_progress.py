#!/usr/bin/env python3
"""从本地已验证证据生成可提交的中文进度摘要。

CPU-only reader: it consumes already-written JSON/Markdown evidence and does not
run GPU, Docker, model loading, network, publication, or Git actions.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Fidelity = "fidelity_bf16_exact"
Practical = "practical_disclosed_approx"
R8_MATCHED_TERMINAL_RUN_PREFIX = "sol_attn_h3_matched_retest_r8_n3_"
R8_MATCHED_TERMINAL_RECHECK_PREFIX = "r8_matched_retest_terminal_recheck_"
R8_MATCHED_NONTERMINAL_PREFIXES = (
    "r8_matched_retest_inspection_",
    "r8_matched_retest_nonterminal_inspection_",
)
R8_MATCHED_TERMINAL_FILES = (
    "decision.json",
    "RUN_REPORT.md",
    "timing_summary.json",
    "quality_proxy_comparison.json",
    "resource_summary.json",
)
R8_FORMAL_N10_PREFIX = "sol_attn_h3_formal_n10_r8_n"
R8_FORMAL_N10_TERMINAL_FILES = (
    "formal_n10_decision.json",
    "RUN_REPORT.md",
    "formal_n10_summary.json",
    "timing_summary.json",
    "quality_proxy_comparison.json",
    "resource_summary.json",
)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def text(path: Path, default: str = "待定") -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return value or default
    except Exception:
        return default


def latest_dir(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted(path for path in parent.glob(f"{prefix}*") if path.is_dir())
    return candidates[-1] if candidates else None


def rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def latest_terminal_decision_dir(evidence: Path) -> Path | None:
    parent = evidence / "sol_engine_port"
    if not parent.is_dir():
        return None
    for path in sorted((p for p in parent.glob(f"{R8_MATCHED_TERMINAL_RUN_PREFIX}*") if p.is_dir()), reverse=True):
        if (path / "decision.json").is_file():
            return path
    return None


def formal_pair_completed(pair_dir: Path) -> bool:
    if (pair_dir / "decision.json").is_file():
        return True
    return text(pair_dir.parent / f"{pair_dir.name}.exit_code", "") == "0"


def collect_r8_formal_n10(evidence: Path) -> dict[str, Any]:
    parent = evidence / "sol_engine_port"
    formal_dir = latest_dir(parent, R8_FORMAL_N10_PREFIX)
    if formal_dir is None:
        return {}
    status_path = formal_dir / "formal_n10_supervisor_status.json"
    decision_path = formal_dir / "formal_n10_decision.json"
    report_path = formal_dir / "RUN_REPORT.md"
    supervisor = load(status_path)
    pair_dirs = sorted(path for path in formal_dir.glob("pair[0-9][0-9]") if path.is_dir())
    completed_pair_dirs = [path for path in pair_dirs if formal_pair_completed(path)]
    terminal_artifacts = {name: (formal_dir / name).is_file() for name in R8_FORMAL_N10_TERMINAL_FILES}
    if decision_path.is_file():
        decision = load(decision_path)
        gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
        gpu = decision.get("same_baseline_physical_gpu_evidence") if isinstance(decision.get("same_baseline_physical_gpu_evidence"), dict) else {}
        return {
            "status_kind": "terminal",
            "status": decision.get("formal_classification", "待定"),
            "reason": decision.get("reason", "待定"),
            "source_run_dir": rel(formal_dir, evidence),
            "decision_evidence": rel(decision_path, evidence),
            "report_evidence": rel(report_path, evidence) if report_path.is_file() else None,
            "summary_evidence": rel(formal_dir / "formal_n10_summary.json", evidence) if (formal_dir / "formal_n10_summary.json").is_file() else None,
            "terminal_artifacts_present": terminal_artifacts,
            "requested_pairs": gates.get("requested_pairs", decision.get("requested_pairs", supervisor.get("n_pairs", 10))),
            "started_pairs": len(pair_dirs),
            "completed_pairs": gates.get("completed_pairs", decision.get("completed_pairs", len(completed_pair_dirs))),
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "failed_gates": decision.get("failed_gates", []),
            "supervisor_status": supervisor.get("status", "待定"),
            "same_expected_gpu": gpu.get("same_expected_gpu", decision.get("same_expected_gpu")),
        }
    status = "incomplete_formal_n10_running_no_terminal_decision" if supervisor.get("status") == "running" else "incomplete_formal_n10_no_terminal_decision"
    return {
        "status_kind": "nonterminal",
        "status": status,
        "reason": "formal N>=10 run has no formal_n10_decision.json/RUN_REPORT terminal artifacts; do not promote or claim speedup",
        "source_run_dir": rel(formal_dir, evidence),
        "decision_evidence": rel(status_path, evidence) if status_path.is_file() else rel(formal_dir, evidence),
        "requested_pairs": supervisor.get("n_pairs", 10),
        "started_pairs": len(pair_dirs),
        "completed_pairs": len(completed_pair_dirs),
        "supervisor_status": supervisor.get("status", "待定"),
    }


def collect_r8_matched_retest(evidence: Path) -> dict[str, Any]:
    """Return terminal r8 matched-route evidence before older inspections.

    The supervisor may leave older read-only nonterminal probes behind.  Once the
    CPU-only finalizer/recheck writes decision artifacts, those terminal facts are
    the authoritative progress surface.  This function never interprets the N=3
    route gate as formal speedup, BF16 fidelity, release, or quality equivalence.
    """
    delivery = evidence / "delivery"
    terminal_dir = latest_terminal_decision_dir(evidence)
    recheck_dir = latest_dir(delivery, R8_MATCHED_TERMINAL_RECHECK_PREFIX)
    recheck_path = recheck_dir / "summary.json" if recheck_dir is not None else None
    if terminal_dir is not None:
        decision_path = terminal_dir / "decision.json"
        decision = load(decision_path)
        gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
        terminal_artifacts = {name: (terminal_dir / name).is_file() for name in R8_MATCHED_TERMINAL_FILES}
        return {
            "status_kind": "terminal",
            "classification": decision.get("classification", "待定"),
            "reason": decision.get("reason", "待定"),
            "source_run_dir": rel(terminal_dir, evidence),
            "decision_evidence": rel(decision_path, evidence),
            "terminal_recheck_evidence": rel(recheck_path, evidence) if recheck_path is not None and recheck_path.is_file() else None,
            "terminal_artifacts_present": terminal_artifacts,
            "requested_pairs": gates.get("requested_pairs", decision.get("requested_pairs")),
            "completed_pairs": gates.get("completed_pairs", decision.get("completed_pairs")),
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "failed_gates": decision.get("failed_gates", []),
            "n10_recommendation": "proceed_to_formal_n10_candidate" if decision.get("proceed_to_n10_recommended") is True else "do_not_promote_without_new_evidence",
            "not_formal_n10": decision.get("not_formal_n10"),
            "not_fidelity_or_performance_claim": decision.get("not_fidelity_or_performance_claim"),
        }
    if recheck_path is not None and recheck_path.is_file():
        recheck = load(recheck_path)
        return {
            "status_kind": "terminal",
            "classification": recheck.get("classification", "待定"),
            "reason": recheck.get("reason", "待定"),
            "source_run_dir": recheck.get("source_run_dir"),
            "decision_evidence": None,
            "terminal_recheck_evidence": rel(recheck_path, evidence),
            "requested_pairs": recheck.get("requested_pairs"),
            "completed_pairs": recheck.get("completed_pairs"),
            "median_http_time_improvement_pct": recheck.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": recheck.get("timing_threshold_pct"),
            "failed_gates": recheck.get("failed_gates", []),
            "n10_recommendation": "proceed_to_formal_n10_candidate" if recheck.get("proceed_to_n10_recommended") is True else "do_not_promote_without_new_evidence",
            "not_formal_n10": recheck.get("not_formal_n10"),
            "not_fidelity_or_performance_claim": recheck.get("not_fidelity_or_performance_claim"),
        }

    latest_nonterminal: Path | None = None
    for prefix in R8_MATCHED_NONTERMINAL_PREFIXES:
        candidate = latest_dir(delivery, prefix)
        if candidate is not None and (latest_nonterminal is None or candidate.name > latest_nonterminal.name):
            latest_nonterminal = candidate
    if latest_nonterminal is None:
        return {}
    preferred = latest_nonterminal / "r8_matched_retest_nonterminal_inspection.json"
    json_files = [preferred] if preferred.is_file() else sorted(latest_nonterminal.glob("*.json"))
    if not json_files:
        return {}
    matched = load(json_files[0])
    return {
        "status_kind": "nonterminal",
        "classification": matched.get("classification", "待定"),
        "reason": matched.get("reason", "待定"),
        "source_run_dir": matched.get("source_run_dir"),
        "evidence_path": rel(json_files[0], evidence),
        "n10_recommendation": matched.get("n10_recommendation", "待定"),
    }


def f(value: Any, n: int = 3) -> str:
    return f"{value:.{n}f}" if isinstance(value, (int, float)) else "待定"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evidence = args.evidence_root
    baseline = load(evidence / "baseline_a6000/baseline_certification.json")
    warm = baseline.get("warm_requests_primary_denominator", {})

    timing_id = text(evidence / "turbo_merged/timing_repeats/LATEST_RUN_ID", "")
    timing = load(evidence / "turbo_merged/timing_repeats" / timing_id / "timing_summary.json") if timing_id else {}
    schedules = timing.get("schedules", {})

    dlo_state = evidence / "dlo_autotune/detached_continuation"
    dlo_id = text(dlo_state / "candidate50_run_id.txt", "")
    dlo = load(evidence / "dlo_autotune/runs" / dlo_id / "candidate50_summary.json") if dlo_id else {}

    delivery = evidence / "delivery"
    r8_dir = latest_dir(delivery, "r8_sol_attn_cpu_ingest_")
    r8 = load(r8_dir / "r8_terminal_classification.json") if r8_dir else {}
    r8_tele = r8.get("telemetry", {}) if isinstance(r8.get("telemetry", {}), dict) else {}
    r8_res = r8.get("resource_summary", {}) if isinstance(r8.get("resource_summary", {}), dict) else {}

    matched = collect_r8_matched_retest(evidence)
    formal = collect_r8_formal_n10(evidence)

    lifecycle_dir = latest_dir(delivery, "local_lifecycle_clean_room_")
    lifecycle = load(lifecycle_dir / "lifecycle/stages/05_lifecycle_summary.json") if lifecycle_dir else {}
    model_prepare = load(lifecycle_dir / "lifecycle/stages/02_model_prepare.json") if lifecycle_dir else {}

    cpu_gate_dir = latest_dir(delivery, "final_cpu_static_gate_")
    cpu_gate_summary = text(cpu_gate_dir / "summary.txt", "") if cpu_gate_dir else ""
    cpu_gate_status = "pass" if "status=pass" in cpu_gate_summary else "待定"
    decisive_gate_dir = latest_dir(delivery, "final_decisive_export_audit_")
    decisive_gate = load(decisive_gate_dir / "summary.json") if decisive_gate_dir else {}
    decisive_gate_status = decisive_gate.get("status", "待定")
    formal_sync_gate_dir = latest_dir(delivery, "formal_n10_cpu_sync_export_audit_")
    formal_sync_gate = load(formal_sync_gate_dir / "summary.json") if formal_sync_gate_dir else {}
    formal_sync_gate_status = formal_sync_gate.get("status", "not_available")
    formal_sync_reviewer_status = str(formal_sync_gate.get("reviewer_status", ""))
    formal_sync_reviewer_accepted = "accepted" in formal_sync_reviewer_status or formal_sync_reviewer_status in {
        "passed",
        "reviewer_passed",
        "independent_reviewer_passed",
    }
    formal_sync_private_synced = bool(formal_sync_gate.get("push_performed"))
    final_gates_pass = cpu_gate_status == "pass" and decisive_gate_status == "pass" and formal_sync_gate_status in {"not_available", "pass"}
    matched_terminal = matched.get("status_kind") == "terminal"
    formal_accepted = formal.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
    formal_incomplete = str(formal.get("status", "")).startswith("incomplete_")
    sol_matched_boundary = (
        "- Sol-Attn r8 formal N>=10 matched-workload gate已终端接受：仅限formal 5-step Sol-Attn opt-in lane；不是BF16 fidelity、release或人类听感/语义质量认证。"
        if formal_accepted
        else (
            "- Sol-Attn r8已有terminal N=3 matched-workload route gate；它只支持未来formal N>=10候选推荐，不是正式speedup、BF16 fidelity、release或质量等价声明。"
            if matched_terminal
            else "- Sol-Attn r8只清除了5-step metadata sparse路径执行gate；matched-workload correctness/quality/performance仍需终端证据。"
        )
    )
    matched_next_step = (
        (
            "formal N>=10已终端接受且独立Reviewer已通过；post-review private main同步已完成，后续不要重复formal run或扩大为BF16/质量/release声明。"
            if formal_sync_private_synced
            else "formal N>=10已终端接受且独立Reviewer已通过；下一步只是在fresh audit pass后做非强制private main同步，不要重复formal run或扩大为BF16/质量/release声明。"
        )
        if formal_accepted and formal_sync_reviewer_accepted
        else (
            "formal N>=10已终端接受；下一步是同步report/export/audit证据并请求独立Reviewer，不要自行push或扩大为BF16/质量/release声明。"
            if formal_accepted
            else (
                "formal N>=10已发现非终端/不完整artifact；先诊断/补齐terminal evidence，不得重复启动同一formal run。"
                if formal_incomplete
                else (
                    "如获授权，下一步是formal N>=10 matched-workload gate；不得把N=3 route gate写成正式speedup、BF16 fidelity或质量等价。"
                    if matched_terminal
                    else "等待/复核r8 matched retest终端artifact；未终端前不晋级N>=10或写speedup。"
                )
            )
        )
    )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# MiniMax-H3 A6000 验证进度",
        "",
        f"> 更新时间：{now}",
        "> 仅汇总已落盘证据；实验中或缺失结果明确标记，不估算。",
        "",
        "## 固定工作负载",
        "",
        "- 单张RTX A6000 48GB；完整FL2VA；1344×768；5.166667秒；124帧；24FPS；32kHz立体声音频。",
        "",
        "## 已认证性能",
        "",
        "| 路径 | Steps | N | Median(s) | Speedup |",
        "|---|---:|---:|---:|---:|",
        f"| BF16 fidelity | 50 | {warm.get('n', '待定')} | {f(warm.get('median_s'))} | 1.000× |",
        f"| Turbo 8-step | 8 | {schedules.get('8', {}).get('n', '待定')} | {f(schedules.get('8', {}).get('median_s'))} | {f(schedules.get('8', {}).get('speedup_vs_same_gpu3_bf16_warm_n10_median'))}× |",
        f"| Turbo 4-step | 4 | {schedules.get('4', {}).get('n', '待定')} | {f(schedules.get('4', {}).get('median_s'))} | {f(schedules.get('4', {}).get('speedup_vs_same_gpu3_bf16_warm_n10_median'))}× |",
        "",
        "## DLO",
        "",
        f"- 状态：{text(dlo_state / 'status.txt')}；resident_layers={dlo.get('resident_layers', '待定')}；warm={f(dlo.get('warm_latency_s'))}秒；改善={f(dlo.get('single_warm_improvement_percent'))}%；正式N10建议={dlo.get('formal_n10_promotion_recommended', '待定')}。",
        "- 当前边界：0.456%候选改善低于baseline warm CV 0.837%，没有正式DLO N10晋级证据。",
        "",
        "## Sol-Attn r8",
        "",
        f"- 5-step gate分类：{r8.get('classification', '待定')}；final={r8.get('final_pass_fail', '待定')}；release_manifest_eligible={r8.get('release_manifest_eligible', '待定')}。",
        f"- sparse_candidate_calls={r8_tele.get('sparse_candidate_calls', '待定')}；sparse_calls={r8_tele.get('sparse_calls', '待定')}；dense_calls={r8_tele.get('dense_calls', '待定')}；fallback_calls={r8_tele.get('fallback_calls', '待定')}；declines={r8_tele.get('decline_reasons', {})}。",
        f"- density_samples={r8_tele.get('density_sample_count', '待定')}；materialized_copy={r8_tele.get('materialized_copy_calls', '待定')}次/{r8_tele.get('materialized_copy_bytes', '待定')} bytes；peak_memory={r8_res.get('peak_gpu_memory_mib', '待定')} MiB；peak_temp={r8_res.get('peak_temperature_c', '待定')}C；peak_power={r8_res.get('peak_power_w', '待定')}W。",
        "- 边界：这是5-step metadata-plumbing sparse-execution diagnostic candidate pass，不是speedup、N10、BF16 fidelity、release或质量等价声明。",
    ]
    if matched_terminal:
        lines.extend(
            [
                f"- matched retest route-decision：{matched.get('classification', '待定')}；completed_pairs={matched.get('completed_pairs', '待定')}/{matched.get('requested_pairs', '待定')}；median_http_time_improvement={f(matched.get('median_http_time_improvement_pct'))}%；threshold={f(matched.get('timing_threshold_pct'))}%；failed_gates={matched.get('failed_gates', '待定')}；n10_recommendation={matched.get('n10_recommendation', '待定')}。",
                f"- matched retest evidence：decision={matched.get('decision_evidence') or '待定'}；terminal_recheck={matched.get('terminal_recheck_evidence') or '待定'}。",
                "- matched retest边界：这是terminal N=3 route gate，已导向后续formal N>=10终端接受；N=3本身仍不是正式speedup、BF16 fidelity、release或质量等价声明，也不替代人类听感。" if formal_accepted else "- matched retest边界：这是terminal N=3 route gate，仅建议未来formal N>=10候选；不是正式speedup、BF16 fidelity、release或质量等价声明，也不替代人类听感。",
                f"- matched retest reason：{matched.get('reason', '待定')}",
                "",
            ]
        )
    elif matched:
        lines.extend(
            [
                f"- matched retest route-decision：{matched.get('classification', '待定')}；n10_recommendation={matched.get('n10_recommendation', '待定')}；reason={matched.get('reason', '待定')}",
                "",
            ]
        )
    else:
        lines.append("")

    if formal:
        lines.extend(
            [
                f"- formal N>=10：{formal.get('status', '待定')}；completed_pairs={formal.get('completed_pairs', '待定')}/{formal.get('requested_pairs', '待定')}；median_http_time_improvement={f(formal.get('median_http_time_improvement_pct'))}%；threshold={f(formal.get('timing_threshold_pct'))}%；same_expected_gpu={formal.get('same_expected_gpu', '待定')}；supervisor_status={formal.get('supervisor_status', '待定')}。",
                f"- formal N>=10 evidence：decision={formal.get('decision_evidence') or '待定'}；summary={formal.get('summary_evidence') or '待定'}；terminal_artifacts={formal.get('terminal_artifacts_present', '待定')}。",
                "- formal N>=10边界：仅限formal 5-step Sol-Attn opt-in matched-workload lane；不是BF16 fidelity、release或人类听感/语义质量认证。" if formal_accepted else "- formal N>=10边界：未终端/未接受前不得称为speedup、BF16 fidelity、release或质量等价。",
                "",
            ]
        )

    gate_next_step = (
        (
            "CPU/static、fixture、Turbo dry-run、strict aggregation、export和publication audit gate已通过；独立Reviewer已通过且private main非强制同步已完成。"
            if formal_sync_private_synced
            else "CPU/static、fixture、Turbo dry-run、strict aggregation、export和publication audit gate已通过；独立Reviewer已通过，剩余动作是fresh audit后的非强制private main同步。"
        )
        if final_gates_pass and formal_sync_reviewer_accepted
        else (
            "CPU/static、fixture、Turbo dry-run、strict aggregation、export和publication audit gate已通过；下一步是独立Reviewer，而不是自行push。"
            if final_gates_pass
            else "运行CPU/static、fixture、Turbo dry-run、strict aggregation和publication audit最终gate。"
        )
    )
    lines.extend(
        [
            "## Clean-room one-command local lifecycle",
            "",
            f"- 状态：{lifecycle.get('status', '待定')}；publication_audit={lifecycle.get('publication_audit_status', '待定')}；local资源只读检查={model_prepare.get('local_non_symlink_file_count', '待定')} files/{model_prepare.get('local_total_bytes', '待定')} bytes。",
            "- 边界：packaging/deployment evidence only；未启动容器、未加载/修改权重、未运行GPU inference/媒体生成，不产生速度/质量/保真声明。",
            "",
            "## Final delivery gates",
            "",
            f"- CPU/static gate：{cpu_gate_status}；evidence={cpu_gate_dir.relative_to(evidence).as_posix() if cpu_gate_dir else '待定'}。",
            f"- strict aggregation/export/publication audit：{decisive_gate_status}；export_file_count={decisive_gate.get('export_file_count', '待定')}；publication_issue_count={decisive_gate.get('publication_issue_count', '待定')}；evidence={decisive_gate_dir.relative_to(evidence).as_posix() if decisive_gate_dir else '待定'}。",
            f"- formal N10 report-sync export/publication audit：{formal_sync_gate_status}；export_file_count={formal_sync_gate.get('export_file_count', '待定')}；publication_issue_count={formal_sync_gate.get('publication_issue_count', '待定')}；reviewer_status={formal_sync_gate.get('reviewer_status', '待定')}；push_performed={formal_sync_gate.get('push_performed', False)}；evidence={formal_sync_gate_dir.relative_to(evidence).as_posix() if formal_sync_gate_dir else '待定'}。",
            "- 边界：这些是CPU/static/export/audit gate，不产生GPU、Docker-run、model-load、保真或质量新声明；formal N10速度候选只来自已落盘GPU证据，不由CPU sync gate新产生。",
            "",
            "## 当前边界",
            "",
            f"- BF16 fidelity lane仅包含baseline；Turbo属于{Practical}，不得混入无损结论。",
            sol_matched_boundary,
            "- Turbo自动结构/音频指标已完成；真实人工听感仍需操作者本人完成，agent不得冒充。",
            "- DMD/DMD2在无合法可复现H3 recipe/checkpoint时保持blocked。",
            "",
            "## 下一步",
            "",
            f"1. {matched_next_step}",
            f"2. {gate_next_step}",
            (
                "3. sanitized release tree已在Reviewer通过和fresh audit pass后提交/同步到既有Private GitHub main。"
                if formal_sync_private_synced
                else (
                    "3. 独立Reviewer已通过；fresh audit pass后把sanitized release tree提交并非强制push到既有Private GitHub main。"
                    if formal_sync_reviewer_accepted
                    else "3. 独立Reviewer通过后，才把sanitized release tree提交并push到既有Private GitHub main。"
                )
            ),
            "4. 如果只剩人类主观听感，保留operator listening gate和文件映射。",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
