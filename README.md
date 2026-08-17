<h1 align="center">MiniMax-H3 on a Single RTX A6000</h1>

<p align="center">
  <strong>完整 FL2VA，一张 48GB A6000，直接生成 1344×768 同步音视频</strong><br>
  <sub>6.159× practical default · 11.978× fastest measured lane · 30s formal +4.326% · 60s E2E demonstrated</sub>
</p>

<p align="center">
  <a href="CURRENT_WORK.md"><strong>当前工作</strong></a> ·
  <a href="RESEARCH_DASHBOARD.md"><strong>研究看板</strong></a> ·
  <a href="benchmark_contract/v1/README.md"><strong>Benchmark Contract</strong></a> ·
  <a href="README_EN.md">English</a> ·
  <a href="#实际生成效果">观看效果</a> ·
  <a href="#5-分钟快速开始">快速开始</a> ·
  <a href="#一眼看懂量化成绩">量化成绩</a> ·
  <a href="technical_report/minimax_h3_a6000_performance.md">完整报告</a>
</p>

<p align="center">
  <img alt="GPU RTX A6000" src="https://img.shields.io/badge/GPU-1%C3%97RTX%20A6000%2048GB-76B900?logo=nvidia&logoColor=white">
  <img alt="Turbo speed" src="https://img.shields.io/badge/Turbo%208--step-6.159%C3%97-00A67E">
  <img alt="Fastest measured lane" src="https://img.shields.io/badge/Turbo%204--step-11.978%C3%97-F59F00">
  <img alt="30 second result" src="https://img.shields.io/badge/30s%20formal-%2B4.326%25-4C8BF5">
  <img alt="60 second E2E" src="https://img.shields.io/badge/60s%20final--AV-E2E%20complete-845EF7">
  <img alt="Resolution 1344x768" src="https://img.shields.io/badge/output-1344%C3%97768%20%2B%20stereo-E64980">
  <img alt="License Apache 2.0" src="https://img.shields.io/badge/code-Apache--2.0-blue">
</p>

> [!IMPORTANT]
> **当前最强正式长视频结果：**单张 RTX A6000 上，30 秒 final-AV `r10_adaptive_tau1_5_step3_diag` matched formal N=10 的 warm E2E median 为 **1333.575 秒**，对照 retained r9 为 **1394.006 秒**，改善 **4.326%**。60 秒完整链路也已真实跑通，r10 N=1 为 **2682.008 秒**、r9 为 **2802.991 秒**；这个 **4.316% 仅是单样本研发信号**，因自动质量红旗而没有晋级为 formal speedup。

**不需要 A100/H100，也没有把多卡服务器结果冒充桌面结果。** 本项目在一张真实的 **NVIDIA RTX A6000 48GB（SM86）**上跑通完整 MiniMax-H3 FL2VA，并交付 BF16 基线、Turbo practical 路线、真实 Sol-Attn sparse execution、30/60 秒 final-AV、部署脚本、测试和可审计证据。所有加速都使用同一物理 GPU 的冻结对照；没有通过质量门禁的候选会公开标记为 rejected，而不会包装成结果。

<p align="center">
  <a href="examples/a6000-turbo-8step-niulai-inspired/niulai-inspired-forest-awakening-turbo-8step.mp4">
    <img src="examples/a6000-turbo-8step-niulai-inspired/hero-frame.jpg" alt="Forest Awakening — MiniMax-H3 FL2VA generated on one RTX A6000">
  </a><br>
  <strong>最新 FL2VA 主视觉 · 《雨后新生》</strong><br>
  <sub>根据操作者提供参考图生成的非官方原创镜头 · 点击图片播放完整视频</sub>
</p>

## 一眼看懂量化成绩

| 能力 | 实测结果 | 你可以怎么理解 |
|---|---:|---|
| BF16 fidelity 短片 | **1792.202 秒**，warm N=10 median | 同卡保真分母 |
| Turbo 8-step 短片 | **290.998 秒，6.159×** | 当前推荐 practical 默认路线 |
| Turbo 4-step 短片 | **149.619 秒，11.978×** | 更快，但保留已知视觉失败样本 |
| Sol-Attn r8 短片 | **15.203%** median HTTP-time improvement，10/10 pairs | 真实 sparse calls，不是 toy benchmark |
| 30 秒 r10 final-AV | **1333.575 vs 1394.006 秒，+4.326%** | 已通过独立 Reviewer 的 formal N=10 结果 |
| 60 秒 r10 final-AV | **2682.008 vs 2802.991 秒，N=1 +4.316% 信号** | 完整 1440 帧/立体声演示；描述性、未晋级 |
| 最新 VAE cap=4 候选 | **1302.506 vs 1331.377 秒，N=1 +2.168% 信号** | VAE 快约 14.1%，但质量门禁失败，已拒绝 |

> **口径边界：**30/60 秒输出采用 extension/chunked 生产方式，不是原生长上下文；Turbo、Sol-Attn 和 VAE batching 属于 `practical_disclosed_approx`，不能写成 BF16 fidelity。N=1 数字只用于决定是否继续实验，不是正式 speedup。

## 从模型到最终音视频

<p align="center">
  <a href="docs/assets/minimax-h3-a6000-pipeline.svg">
    <img src="docs/assets/minimax-h3-a6000-pipeline.svg" alt="MiniMax-H3 single RTX A6000 model-to-verified-audiovisual architecture">
  </a>
</p>

<p align="center">
  <sub>完整 FL2VA → fidelity/practical 分轨 → 单张 A6000 + DLO → guarded r10 Sol-Attn → 5.17s / 30s / 60s 输出 → A/V 组装与验证门禁</sub>
</p>

> 图中的绿色是正式 accepted evidence，琥珀色是 practical 路线，紫色是 descriptive/no-promotion，珊瑚色表示 rejected 但保留证据。30/60 秒始终标记为 extension/chunked，不冒充 native long context。

---

## 实际生成效果

### 主视觉：从参考图到原创动态镜头

<a href="examples/a6000-turbo-8step-niulai-inspired/niulai-inspired-forest-awakening-turbo-8step.mp4">
  <img src="examples/a6000-turbo-8step-niulai-inspired/hero-frame.jpg" alt="MiniMax-H3 A6000 FL2VA Forest Awakening hero frame">
</a>

<p align="center">
  <strong>《雨后新生 / Forest Awakening》</strong><br>
  <sub>点击主视觉即可播放 1344×768、24 FPS、32 kHz stereo 完整视频</sub>
</p>

<a href="examples/a6000-turbo-8step-niulai-inspired/niulai-inspired-forest-awakening-turbo-8step.mp4">
  <img src="examples/a6000-turbo-8step-niulai-inspired/contact-sheet.jpg" alt="Forest Awakening six-frame generated sequence">
</a>

- **[观看或下载完整 FL2VA 视频](examples/a6000-turbo-8step-niulai-inspired/niulai-inspired-forest-awakening-turbo-8step.mp4)**
- [完整提示词](examples/a6000-turbo-8step-niulai-inspired/prompt.txt) · [生成、择优与验证元数据](examples/a6000-turbo-8step-niulai-inspired/metadata.json)
- 输入：操作者提供的本地《牛来》画面参考；源图 SHA256 已记录但**不在仓库中再分发**
- 生成：3 个 8-step seeds 全部完成后比较，选择 seed 42；没有只展示第一个随机样本
- 实测请求耗时：**305.386 秒（约 5 分 5 秒）**
- 实测资源：峰值显存 **27,410 MiB**、峰值功耗 **301.16 W**、峰值温度 **79°C**
- 输出验证：**124/124 帧**、32 kHz 双声道、每声道 166,912 samples、冻结转场 `<0.05` 计数为 **0**
- 视觉检查：两只牛的配色、角型、左右身份保持稳定；幼苗、手势和渐强体积光形成完整的单镜头叙事

> [!NOTE]
> 这是根据操作者提供参考图生成的**非官方原创演示**，不是电影《牛来》的片段、官方复刻或合作内容。新镜头运动、光影、幼苗叙事和环境音由 MiniMax-H3 生成；自动检查证明视频/音频结构完整，但不冒充操作者的主观听感认证。

### 完全文本生成：轨道船坞

下面的视频同样由本仓库在单张 RTX A6000 上生成，不是引用其他项目的数据或媒体。

<a href="examples/a6000-turbo-8step-sci-fi/orbital-shipyard-turbo-8step.mp4">
  <img src="examples/a6000-turbo-8step-sci-fi/contact-sheet.jpg" alt="MiniMax-H3 A6000 orbital shipyard six-frame preview">
</a>

- **[点击观看或下载 1344×768 科幻生成视频](examples/a6000-turbo-8step-sci-fi/orbital-shipyard-turbo-8step.mp4)**
- [完整提示词](examples/a6000-turbo-8step-sci-fi/prompt.txt) · [运行与校验元数据](examples/a6000-turbo-8step-sci-fi/metadata.json)
- 配置：单张 RTX A6000、Turbo 8-step、seed 42、124 帧、24 FPS、5.1667 秒
- 实际请求耗时：**291.627 秒（约 4 分 52 秒）**
- 输出：H.264 视频 + AAC 32kHz 双声道音频
- 视频 SHA256：`454fceb57b1daf60dc5db1ade9aae295cee2ecd16fd90ab2c2f16c7b626db69a`

**提示词节选：**

```text
A cinematic wide shot inside a colossal orbital shipyard above a luminous blue planet.
A sleek silver exploration starship slowly launches from a glowing circular docking ring
as coherent blue plasma thrusters ignite ...
```

该文件已完成 124 帧全解码、双声道音频解码、非静音/峰值检查、冻结转场 proxy 和六帧人工视觉检查。

---

## A6000 实测性能

固定工作负载：**完整 FL2VA、1344×768、124 帧、24 FPS、5.166667 秒、32kHz 立体声音频**。所有主要 speedup 都使用同一物理 A6000 的 warm BF16 N=10 中位数作为分母。

| 路线 | Steps | 正式 N | Median | 相对 BF16 | 定位 |
|---|---:|---:|---:|---:|---|
| BF16 dense baseline | 50 | 10 | **1792.202 s** | 1.000× | fidelity 分母 |
| **Turbo 8-step** | 8 | 10 | **290.998 s** | **6.159×** | 推荐 practical 默认路线 |
| Turbo 4-step | 4 | 10 | **149.619 s** | **11.978×** | 超快实验路线，质量代价更高 |

### 资源实测

| 路线 | 峰值显存 | 峰值 host memory | 峰值功耗 | 峰值温度 |
|---|---:|---:|---:|---:|
| BF16 baseline | 26,836 MiB | 204.84 GiB | 302.23 W | 84°C |
| Turbo formal timing | 26,836 MiB | 195.16 GiB | 301.08 W | 83°C |
| 本 README 科幻示例会话 | 26,664 MiB | 175.95 GiB | 301.09 W | 83°C |
| 《雨后新生》FL2VA 三候选会话 | 27,410 MiB | 185.95 GiB | 301.16 W | 79°C |

### 怎么理解这些数字

- **8-step** 是目前推荐选项：N=10 稳定达到 6.159×，24-case 质量套件中视觉 12/12 通过。
- **4-step** 更快，但 24-case 套件中视觉 11/12，通过率较低；一个茶壶样本出现明显几何变形，因此不作为默认路线。
- 操作者已完成人工观看/听感审阅并给出整体正向验收；已知 4-step 失败样本仍保留，不因整体评价而删除。
- [公开 review matrix 与六张汇总联系表](technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md)支持 8-step 12/12、4-step 11/12 的视觉计数；24 个原始质量套件 MP4 因发布体积没有全部复制到精简 release tree。
- BF16 与 Turbo 分轨：Turbo 使用静态合并 LoRA，是 `practical_disclosed_approx`，不是无损 BF16 fidelity。

### 30 秒 final-AV extension/chunked formal r10

这是单张 A6000 上的 practical approximate 长视频链路结果，不与上面的 BF16 短片分母交叉计算 speedup。

| 路线 | 生成模式 | 正式 N | Warm E2E median | 对照 | 结论边界 |
|---|---|---:|---:|---:|---|
| `r10_adaptive_tau1_5_step3_diag` | six-chunk `extension` / chunked final-AV | 10 | **1333.575 s** | retained `r9_current_sol_attn` **1394.006 s** | **4.326%** median warm-E2E improvement；720 帧、960,000 samples/channel 完整核算 |

该 r10 结果只说明 matched formal N=10 30 秒 final-AV extension/chunked lane 中，guarded adaptive step-min=3 Sol-Attn 相对 retained r9 有有界 warm-E2E 改善。它明确排除原生长上下文、BF16 fidelity、人类语义/音频质量、产品就绪、公开对比和 SOTA。

### 最新研发前沿：有信号，也有严格拒绝

| 新路线 | Reference → Candidate | 实测信号 | 决策 |
|---|---:|---:|---|
| 60 秒 r10 guarded adaptive | 2802.991 → **2682.008 秒** | N=1 **4.316%** | 完整 1440 帧、1,920,000 samples/channel；自动冻结转场 proxy 红旗，descriptive/no-promotion |
| VAE 全量 spatial tile batching | 1335.018 → **1299.728 秒** | E2E **2.643%**；VAE 202.720 → **167.802 秒** | 初始 N=1 route gate 通过，仍是默认关闭的 practical approximate 候选，未形成 formal 结果 |
| VAE 有界 tile batch cap=4 | 1331.377 → **1302.506 秒** | E2E **2.168%**；VAE 202.424 → **173.886 秒** | subject/background proxy 约下降 10%，严格拒绝，不启动 N=3 |
| DLO async sync-prefetch | 1335.021 → **1334.684 秒** | E2E **0.025%** | 虽将 group-first host enqueue 从 297.799 秒降至 0.0088 秒，但不在关键路径，拒绝 |

- [60 秒 N=1 决策与边界](technical_report/evidence/minimax_h3_desktop/long_video/final-av-60s-r9-current-vs-r10-step3-sol-attn-n1-20260816T164226Z/RUN_REPORT.md)
- [VAE 全量 spatial batching N=1](technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-spatial-tile-batching-n1-lease-20260817T033717Z/RUN_REPORT.md)
- [VAE bounded cap=4 最终拒绝包](technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-tile-batch-cap-4-n1-20260817T110844Z/RUN_REPORT.md)

> **报告新鲜度：**聚合性能报告是 2026-08-16 的 accepted-results 快照；上表中的 2026-08-17 frontier 结果以三个直接链接的 per-run `RUN_REPORT.md` / decision 为准，尚未被包装成新的 accepted 聚合结论。

完整 accepted 统计、CV、质量与资源边界见 [`technical_report/minimax_h3_a6000_performance.md`](technical_report/minimax_h3_a6000_performance.md)。

---

## 已经跑通什么

- [x] 完整 MiniMax-H3 FL2VA checkpoint（约 134.16 GiB）
- [x] 单张 RTX A6000 48GB，容器内只暴露一张 GPU
- [x] 1344×768 / 124 帧 / 24 FPS / 32kHz stereo AV
- [x] BF16 dense warm N=10 baseline
- [x] Turbo 4-step 与 8-step，同卡 paired N=10
- [x] 3 prompts × 4 seeds × 2 schedules 的 24-case Turbo 质量套件
- [x] 可直接观看的 A6000 科幻 T2VA 与《雨后新生》参考图 FL2VA 示例视频
- [x] DLO resident-layer 容量与 50-step 候选评估
- [x] SM86 exact Triton kernel 候选与输出漂移检查
- [x] Sol-Attn r8 真实 H3 metadata plumbing、sparse execution、N=3 route gate 和 formal N=10
- [x] 30 秒 final-AV extension/chunked r10 formal N=10 timing/structural 结果（有界，不是 native/BF16/human-quality/product claim）
- [x] 60 秒 final-AV extension/chunked N=1 完整输出：1440 帧、32 kHz stereo；描述性验收，不宣称 formal speedup
- [x] VAE spatial/bounded tile batching、CUDA Graph、DLO async prefetch、Cache-DiT 与 regional compile 的 fail-closed 实验和负结果
- [x] CPU/static tests、sanitized export、publication audit 与证据报告

---

## 5 分钟快速开始

> “5 分钟”指完成仓库配置和启动准备，不包含约 134 GiB 模型下载、约 20 GB runtime 构建以及实际推理时间。

### 硬件与系统

实测环境：

- Ubuntu 24.04 x86_64
- NVIDIA RTX A6000 48GB，SM86
- NVIDIA driver 580.159.03
- Docker + NVIDIA Container Toolkit
- 约 1 TiB host RAM 的测试主机

建议准备：

- 单张 RTX A6000 48GB；
- **至少 256 GiB host RAM**，因为 BF16 实测峰值约 205 GiB；
- **至少 230 GiB 可用磁盘**，用于官方 FL2VA、Turbo adapter、独立 merged transformer、runtime image 和缓存；
- 已安装 `git`、`docker`、NVIDIA Container Toolkit、Python 3 和 Hugging Face CLI。

### 1. Clone 与登录 Hugging Face

```bash
git clone https://github.com/Argus-AiTeam/minimax-h3-desktop.git
cd minimax-h3-desktop

python3 -m venv .venv
source .venv/bin/activate
pip install 'huggingface_hub[cli]==0.34.4'
hf auth login
```

下载前请阅读并接受 [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，并自行确认许可证、地域和使用场景合规。模型权重不包含在本仓库的 Apache-2.0 代码许可中。

### 2. 先看 dry-run

```bash
make dry-run
```

Dry-run 不调用 Docker、不使用 GPU、不加载模型，也不下载文件。

### 3. 构建固定 runtime

```bash
make runtime
```

等价于：

```bash
bash scripts/build_runtime.sh
```

脚本固定 vLLM-Omni commit `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`，并按上游 `docker/Dockerfile.cuda` 构建本项目实测的 CUDA runtime。构建时间和镜像体积都较大。

### 4. 下载并准备模型

选择一张空闲 GPU；离线 LoRA merge 会使用它，正式推理时仍只暴露这一张卡：

```bash
I_ACCEPT_MINIMAX_H3_LICENSE=YES \
GPU_INDEX=0 \
make models
```

这会：

1. 固定下载 `MiniMaxAI/MiniMax-H3` 的 `FL2VA/*`；
2. 固定下载 Larry Turbo EMA adapter；
3. 校验 adapter SHA256；
4. 在独立目录中执行 FP32 accumulate / BF16 cast 静态 merge；
5. 验证 13 个 transformer shards 和 259 个 LoRA pairs；
6. 保持官方基础权重只读，不在原目录覆盖权重。

如果资产已经存在：

```bash
I_ACCEPT_MINIMAX_H3_LICENSE=YES SKIP_DOWNLOAD=1 GPU_INDEX=0 make models
```

### 5. 生成视频

```bash
I_ACCEPT_MINIMAX_H3_LICENSE=YES \
GPU_INDEX=0 \
SEEDS=42 \
STEPS=8 \
OUTPUT_DIR="$PWD/out/my-first-h3-video" \
make demo
```

自定义提示词：

```bash
printf '%s\n' 'A cinematic robot walking through a neon city in the rain, smooth camera motion, synchronized city ambience, no text, no watermark' > my-prompt.txt

I_ACCEPT_MINIMAX_H3_LICENSE=YES \
GPU_INDEX=0 \
PROMPT_FILE="$PWD/my-prompt.txt" \
OUTPUT_DIR="$PWD/out/neon-robot" \
bash scripts/run_turbo_demo.sh
```

使用本地首帧参考图启用 FL2VA（源图不会复制进模型目录）：

```bash
I_ACCEPT_MINIMAX_H3_LICENSE=YES \
GPU_INDEX=0 \
INPUT_REFERENCE="$PWD/my-reference.png" \
PROMPT_FILE="$PWD/my-prompt.txt" \
OUTPUT_DIR="$PWD/out/reference-animation" \
bash scripts/run_turbo_demo.sh
```

脚本会自动：

- 确认目标 GPU 没有其他 compute process；
- 只向容器暴露一张 GPU；
- 以只读方式挂载模型；
- 在禁网容器内启动本地 vLLM-Omni API；
- 生成 MP4；
- 校验 124 帧、分辨率、32kHz 双声道音频；
- 保存请求耗时、SHA256 和资源监控；
- 退出并删除临时容器。

---

## Sol-Attn：真实 sparse execution，而不是“看起来更快”

早期版本诚实保留了失败路径：

- r6：`unsupported_contiguity`，208 次调用全部 fail-closed；
- r7：packed video layout metadata 没有到达 attention backend，`sparse_calls=0`；
- r8：metadata plumbing 修复后，真实 H3 5-step 中 `sparse_candidate_calls=192`、`sparse_calls=192`、`fallback_calls=0`。

Formal N=10 matched-workload 结果：

| 项目 | 结果 |
|---|---:|
| 完成 pairs | 10/10 |
| HTTP/结构 AV | 全部通过 |
| 中位 HTTP 时间改善 | **15.203%** |
| 晋级门槛 | 3.0% |
| sparse calls / pair | 192 |
| fallback calls | 0 |

该结论只适用于 **5-step Sol-Attn opt-in matched lane**。它不是 50-step BF16 fidelity speedup，也不代表 Turbo、DLO 或完整语义质量等价。实现默认关闭、metadata 缺失时 fail-closed 到 dense。

30 秒 final-AV extension/chunked r10 formal N=10 进一步保留默认关闭的 guarded adaptive Sol-Attn step-min=3：warm E2E median **1333.575 s** vs retained r9 **1394.006 s**，median improvement **4.326%**，10/10 pairs final-AV 完整。这个结论只适用于 matched 30 秒 extension/chunked practical lane，不是 native long context、BF16 fidelity、人类语义/音频质量、产品就绪、公开对比或 SOTA。

后续 `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES` 路线仍默认关闭，只在 captured-metadata、非 Docker、无模型 SM86 replay 中得到保留：candidate total median **144.652 ms** vs current prefix-skip **175.216 ms**，forward pointer subphase **129.850 ms** vs **158.161 ms**，`max_abs_valid=0`，且 replay lanes 中无非预期 materialized copy 次数/字节。这个结果只说明 captured metadata kernel 机制可运行；没有加载 H3 模型、没有启动 Docker，不是 H3 端到端、长视频、BF16 保真、普通电脑、产品加速、公开对比或 SOTA 声明。

---

## 哪些优化被拒绝或暂停

| 路线 | 结论 | 原因 |
|---|---|---|
| DLO RL16 50-step | 不晋级 formal N=10 | 单样本改善 0.456%，低于 baseline CV 0.837% |
| RoPE/all-exact | 拒绝 | 视频/音频输出发生漂移 |
| SwiGLU E2E | 不部署 | 没有可保留的端到端收益 |
| toy Sol-Attn | 拒绝 | sparse microbenchmark 比 dense 慢 |
| DMD/DMD2 | research-only blocked | 没有合法、第一方、可复现的 H3 recipe/checkpoint |
| r11/r12 更激进 adaptive routing | 拒绝 | N=1 仅有 1.135% / 0.358% 信号，objective quality proxy 未通过 |
| Cache-DiT high/high_warmup2 | 拒绝 | 没有实际 cache reuse，warm E2E 无收益或略慢 0.031% |
| VAE CUDA Graph | 拒绝 | bit-exact，但 32.865 → 32.945 秒，略慢 |
| VAE bounded tile batch cap=4 | 拒绝 | E2E 有 2.168% 信号，但 subject/background proxy 超出 5% non-inferiority 门槛 |
| DLO async sync-prefetch | 拒绝 | telemetry 大幅改善，但 E2E 仅 0.025%，低于 1% 晋级门槛 |
| Regional `torch.compile` | no-go | graph breaks/recompiles 后超时，无 candidate 媒体；源码信号模型也低于门槛 |

我们保留负结果，避免只展示成功样本、事后改变门槛，或把 kernel/telemetry microbenchmark 写成完整模型加速。

---

## 仓库结构

```text
examples/                         可直接观看的 A6000 生成示例
scripts/build_runtime.sh          构建固定 vLLM-Omni CUDA runtime
scripts/prepare_models.sh         下载、校验并离线 merge Turbo
scripts/run_turbo_demo.sh         单 A6000 实际视频生成入口
ports/minimax_h3_a6000/           SM86 kernels、Sol-Attn、patch 与测试
runtime/single_a6000_bf16/        固定版本/digest/依赖元数据
technical_report/                 性能、质量、资源和证据报告
schemas/                          运行记录 Schema
Makefile                          dry-run/runtime/models/demo/test/audit 入口
```

模型权重、adapter、Docker layers、缓存、凭据和私有运行日志不会提交到 GitHub。

---

## 验证与开发

```bash
make test
make audit
```

更精确的命令：

```bash
PYTHONPATH=code:ports/minimax_h3_a6000/src \
python3 -m pytest -q tests ports/minimax_h3_a6000/tests

python3 tools/publication_audit.py \
  --root . \
  --max-bytes 15000000 \
  --json
```

---

## 平台边界

- 当前正式目标平台是 **单张 RTX A6000 48GB**；
- 测试主机虽然有 4 张 A6000，但每个正式模型进程只看见一张，结果没有聚合多卡；
- 不能将本仓库数字当作 RTX 5090、DGX Spark、A100、H100 或 8×GB200 结果；
- Turbo 是 practical approximation；BF16 baseline 才属于 fidelity 分母；
- 实际耗时会受 prompt 长度、驱动、温度、存储、host RAM 和后台负载影响。

如果你使用 Apple Silicon，请参见姐妹项目 [`Argus-AiTeam/minimax-h3-mac`](https://github.com/Argus-AiTeam/minimax-h3-mac)。两个仓库共享“真实设备、真实媒体、证据优先”的方法，但代码路径与性能数据完全独立。

---

## 项目与上游

- 本项目：<https://github.com/Argus-AiTeam/minimax-h3-desktop>
- Mac 姐妹项目：<https://github.com/Argus-AiTeam/minimax-h3-mac>
- MiniMax-H3：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- vLLM-Omni：<https://github.com/vllm-project/vllm-omni>
- Turbo LoRA：<https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>
- NVLabs/Sana Sol-Engine：<https://github.com/NVlabs/Sana>

## 致谢

- **MiniMaxAI**：MiniMax-H3 架构与官方权重；
- **vLLM / vLLM-Omni**：CUDA 推理和 omni-modal serving 基础；
- **Larry / community Turbo work**：few-step practical adapter；
- **NVLabs/Sana Sol-Engine**：稀疏 attention 与 kernel 方向的上游研究基础；
- **PyTorch、Triton、Hugging Face 与相关社区项目**。

详见 [`NOTICE`](NOTICE)、[`ports/minimax_h3_a6000/NOTICE`](ports/minimax_h3_a6000/NOTICE) 和 [`ports/minimax_h3_a6000/UPSTREAM.md`](ports/minimax_h3_a6000/UPSTREAM.md)。

---

## License

本仓库原创代码以 [Apache License 2.0](LICENSE) 发布。MiniMax-H3 权重、Turbo adapter、生成内容与所有第三方依赖继续受各自许可证和使用条款约束；本仓库不重新授权这些资产。

<p align="center"><strong>完整 MiniMax-H3，现在可以在一张 RTX A6000 上生成带同步音频的 768p 视频。</strong></p>
