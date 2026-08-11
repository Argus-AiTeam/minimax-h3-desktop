# MiniMax-H3 A6000 验证进度

> 更新时间：2026-08-11T17:35:27+00:00
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

## Sol-Attn

- Supervisor：complete；证据状态：fail_closed_dense_fallback。
- sparse candidates=0；sparse calls=0；dense calls=208；declines={'dense_first_steps': 200, 'missing_h3_hook_metadata:missing_packed_video_layout': 8}。

## 当前边界

- BF16和exact候选属于fidelity；Turbo、Sol-Attn、量化和DMD不得混入无损结论。
- DMD在无合法可复现H3 recipe/checkpoint时保持阻塞。
- 真正non-dry一条命令部署和人工音频/AV同步检查仍在推进。

## 下一步

1. 完成当前Sol-Attn诊断并量化sparse/copy/质量/耗时；
2. 更新用户向README和性能报告；
3. 完成non-dry一键部署与clean-room验证；
4. 通过测试和发布审计后持续提交。
