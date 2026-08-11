#!/usr/bin/env python3
"""从本地已验证证据生成可提交的中文进度摘要。"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    try:
        x=json.loads(path.read_text(encoding='utf-8')); return x if isinstance(x,dict) else {}
    except Exception: return {}

def text(path: Path, default='待定') -> str:
    try: return path.read_text(encoding='utf-8').strip() or default
    except Exception: return default

def f(x: Any, n=3) -> str:
    return f'{x:.{n}f}' if isinstance(x,(int,float)) else '待定'

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    e=a.evidence_root
    b=load(e/'baseline_a6000/baseline_certification.json'); warm=b.get('warm_requests_primary_denominator',{})
    tid=text(e/'turbo_merged/timing_repeats/LATEST_RUN_ID',''); t=load(e/'turbo_merged/timing_repeats'/tid/'timing_summary.json') if tid else {}; ts=t.get('schedules',{})
    ds=e/'dlo_autotune/detached_continuation'; did=text(ds/'candidate50_run_id.txt',''); d=load(e/'dlo_autotune/runs'/did/'candidate50_summary.json') if did else {}
    ss=e/'sol_engine_port/sol_attn_gpu2_supervisor'; sid=text(ss/'latest_run_id',''); sr=e/'sol_engine_port'/sid if sid else Path('/'); sol=load(sr/'sol_attn_diagnostic_status.json') if sid else {}; solstatus=text(ss/'status.txt')
    now=datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines=['# MiniMax-H3 A6000 验证进度','',f'> 更新时间：{now}', '> 仅汇总已落盘证据；实验中或缺失结果明确标记，不估算。','',
    '## 固定工作负载','', '- 单张RTX A6000 48GB；完整FL2VA；1344×768；5.166667秒；124帧；24FPS；32kHz立体声音频。','',
    '## 已认证性能','', '| 路径 | Steps | N | Median(s) | Speedup |','|---|---:|---:|---:|---:|',
    f"| BF16 fidelity | 50 | {warm.get('n','待定')} | {f(warm.get('median_s'))} | 1.000× |",
    f"| Turbo 8-step | 8 | {ts.get('8',{}).get('n','待定')} | {f(ts.get('8',{}).get('median_s'))} | {f(ts.get('8',{}).get('speedup_vs_same_gpu3_bf16_warm_n10_median'))}× |",
    f"| Turbo 4-step | 4 | {ts.get('4',{}).get('n','待定')} | {f(ts.get('4',{}).get('median_s'))} | {f(ts.get('4',{}).get('speedup_vs_same_gpu3_bf16_warm_n10_median'))}× |",'',
    '## DLO','',f"- 状态：{text(ds/'status.txt')}；resident_layers={d.get('resident_layers','待定')}；warm={f(d.get('warm_latency_s'))}秒；改善={f(d.get('single_warm_improvement_percent'))}%；正式N10建议={d.get('formal_n10_promotion_recommended','待定')}。",'',
    '## Sol-Attn','',f'- Supervisor：{solstatus}；证据状态：{sol.get("status","待定")}。',]
    if sol:
        tele=sol.get('telemetry',{})
        lines += [f"- sparse candidates={tele.get('sparse_candidate_calls','待定')}；sparse calls={tele.get('sparse_calls','待定')}；dense calls={tele.get('dense_calls','待定')}；declines={tele.get('decline_reasons',{})}。"]
    lines += ['', '## 当前边界','', '- BF16和exact候选属于fidelity；Turbo、Sol-Attn、量化和DMD不得混入无损结论。','- DMD在无合法可复现H3 recipe/checkpoint时保持阻塞。','- 真正non-dry一条命令部署和人工音频/AV同步检查仍在推进。','', '## 下一步','', '1. 完成当前Sol-Attn诊断并量化sparse/copy/质量/耗时；','2. 更新用户向README和性能报告；','3. 完成non-dry一键部署与clean-room验证；','4. 通过测试和发布审计后持续提交。','']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(lines),encoding='utf-8');return 0
if __name__=='__main__': raise SystemExit(main())
