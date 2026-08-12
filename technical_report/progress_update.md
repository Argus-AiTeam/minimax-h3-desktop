# MiniMax-H3 A6000 验证进度

> 更新时间：2026-08-12T05:35:47+00:00
> 仅汇总已落盘证据；实验中或缺失结果明确标记，不估算。

## 固定工作负载

- 单张RTX A6000 48GB；完整FL2VA；1344×768；5.166667秒；124帧；24FPS；32kHz立体声音频。

## 已认证性能

| 路径 | Steps | N | Median(s) | Speedup |
|---|---:|---:|---:|---:|
| BF16 fidelity | 50 | 10 | 1792.202 | 1.000× |
| Turbo 8-step | 8 | 10 | 290.998 | 6.159× |
| Turbo 4-step | 4 | 10 | 149.619 | 11.978× |

## DLO

- 状态：dlo_candidate50_complete；resident_layers=16；warm=1784.069秒；改善=0.456%；正式N10建议=False。
- 当前边界：0.456%候选改善低于baseline warm CV 0.837%，没有正式DLO N10晋级证据。

## Sol-Attn r8

- 5-step gate分类：sparse_runtime_valid_5step_diagnostic；final=candidate_pass；release_manifest_eligible=False。
- sparse_candidate_calls=192；sparse_calls=192；dense_calls=16；fallback_calls=0；declines={'dense_first_layers': 8, 'non_h3_dit_attention_prefix': 8}。
- density_samples=192；materialized_copy=192次/105344139264 bytes；peak_memory=27354.0 MiB；peak_temp=84.0C；peak_power=299.88W。
- 边界：这是5-step metadata-plumbing sparse-execution diagnostic candidate pass，不是speedup、N10、BF16 fidelity、release或质量等价声明。
- matched retest route-decision：proceed_to_formal_n10_candidate；completed_pairs=3/3；median_http_time_improvement=14.782%；threshold=3.000%；failed_gates=[]；n10_recommendation=proceed_to_formal_n10_candidate。
- matched retest evidence：decision=sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json；terminal_recheck=delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json。
- matched retest边界：这是terminal N=3 route gate，仅建议未来formal N>=10候选；不是正式speedup、BF16 fidelity、release或质量等价声明，也不替代人类听感。
- matched retest reason：bounded matched retest passed correctness, sparse-runtime, resource, quality-proxy, and timing gates

## Clean-room one-command local lifecycle

- 状态：pass；publication_audit=pass；local资源只读检查=81 files/144051182625 bytes。
- 边界：packaging/deployment evidence only；未启动容器、未加载/修改权重、未运行GPU inference/媒体生成，不产生速度/质量/保真声明。

## Final delivery gates

- CPU/static gate：pass；evidence=delivery/final_cpu_static_gate_20260812T033107Z。
- strict aggregation/export/publication audit：pass；export_file_count=82；publication_issue_count=0；evidence=delivery/final_decisive_export_audit_20260812T025605Z。
- 边界：这些是CPU/static/export/audit gate，不产生GPU、Docker-run、model-load、速度、保真或质量新声明。

## 当前边界

- BF16 fidelity lane仅包含baseline；Turbo属于practical_disclosed_approx，不得混入无损结论。
- Sol-Attn r8已有terminal N=3 matched-workload route gate；它只支持未来formal N>=10候选推荐，不是正式speedup、BF16 fidelity、release或质量等价声明。
- Turbo自动结构/音频指标已完成；真实人工听感仍需操作者本人完成，agent不得冒充。
- DMD/DMD2在无合法可复现H3 recipe/checkpoint时保持blocked。

## 下一步

1. 如获授权，下一步是formal N>=10 matched-workload gate；不得把N=3 route gate写成正式speedup、BF16 fidelity或质量等价。
2. CPU/static、fixture、Turbo dry-run、strict aggregation、export和publication audit gate已通过；下一步是独立Reviewer，而不是自行push。
3. 独立Reviewer通过后，才把sanitized release tree提交并push到既有Private GitHub main。
4. 如果只剩人类主观听感，保留operator listening gate和文件映射。
