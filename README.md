<h1 align="center">MiniMax-H3 on a Single RTX A6000</h1>

<p align="center">
  <strong>完整 FL2VA · 1344×768 · 同步视频与立体声音频 · Turbo 最高 11.98× · Sol-Attn N=10</strong>
</p>

<p align="center">
  <a href="README_EN.md">English</a> ·
  <a href="#实际生成效果">生成效果</a> ·
  <a href="#5-分钟快速开始">快速开始</a> ·
  <a href="#a6000-实测性能">实测性能</a> ·
  <a href="technical_report/minimax_h3_a6000_performance.md">完整报告</a>
</p>

<p align="center">
  <img alt="GPU RTX A6000" src="https://img.shields.io/badge/GPU-RTX%20A6000%2048GB-76B900?logo=nvidia&logoColor=white">
  <img alt="CUDA SM86" src="https://img.shields.io/badge/CUDA-SM86-76B900?logo=nvidia&logoColor=white">
  <img alt="Resolution 1344x768" src="https://img.shields.io/badge/output-1344%C3%97768%20%2B%20stereo%20audio-4C8BF5">
  <img alt="License Apache 2.0" src="https://img.shields.io/badge/code-Apache--2.0-blue">
</p>

> **本仓库的 A6000 适配、长时间实验、性能验证、失败诊断、Reviewer 审核和文档整理主要由 [Argus](https://github.com/lbx154/Argus) 自主完成，并由 Argus-AiTeam 维护。**
>
> Argus 在这个项目中持续读取代码、修改运行时、调度单卡实验、分析真实音视频结果，并从 r6/r7 的 fail-closed 失败迭代到 r8 的真实 sparse execution 与 formal N=10 验收。

**不需要 A100/H100，也没有把多卡服务器结果冒充桌面结果。** 本项目在一张真实的 **NVIDIA RTX A6000 48GB（SM86）**上跑通完整 MiniMax-H3 FL2VA，并提供可复现的 BF16 baseline、Turbo practical 路线、默认关闭的 Sol-Attn 实验实现、部署脚本、测试和原始证据摘要。

---

## 实际生成效果

下面的视频是本仓库使用单张 RTX A6000 新生成的 **Turbo 8-step practical** 示例。

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

### 怎么理解这些数字

- **8-step** 是目前推荐选项：N=10 稳定达到 6.159×，24-case 质量套件中视觉 12/12 通过。
- **4-step** 更快，但 24-case 套件中视觉 11/12，通过率较低；一个茶壶样本出现明显几何变形，因此不作为默认路线。
- 操作者已完成人工观看/听感审阅并给出整体正向验收；已知 4-step 失败样本仍保留，不因整体评价而删除。
- BF16 与 Turbo 分轨：Turbo 使用静态合并 LoRA，是 `practical_disclosed_approx`，不是无损 BF16 fidelity。

完整统计、CV、质量与资源边界见 [`technical_report/minimax_h3_a6000_performance.md`](technical_report/minimax_h3_a6000_performance.md)。

---

## 已经跑通什么

- [x] 完整 MiniMax-H3 FL2VA checkpoint（约 134.16 GiB）
- [x] 单张 RTX A6000 48GB，容器内只暴露一张 GPU
- [x] 1344×768 / 124 帧 / 24 FPS / 32kHz stereo AV
- [x] BF16 dense warm N=10 baseline
- [x] Turbo 4-step 与 8-step，同卡 paired N=10
- [x] 3 prompts × 4 seeds × 2 schedules 的 24-case Turbo 质量套件
- [x] 可直接观看的 A6000 科幻示例视频
- [x] DLO resident-layer 容量与 50-step 候选评估
- [x] SM86 exact Triton kernel 候选与输出漂移检查
- [x] Sol-Attn r8 真实 H3 metadata plumbing、sparse execution、N=3 route gate 和 formal N=10
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

---

## 哪些优化被拒绝或暂停

| 路线 | 结论 | 原因 |
|---|---|---|
| DLO RL16 50-step | 不晋级 formal N=10 | 单样本改善 0.456%，低于 baseline CV 0.837% |
| RoPE/all-exact | 拒绝 | 视频/音频输出发生漂移 |
| SwiGLU E2E | 不部署 | 没有可保留的端到端收益 |
| toy Sol-Attn | 拒绝 | sparse microbenchmark 比 dense 慢 |
| DMD/DMD2 | research-only blocked | 没有合法、第一方、可复现的 H3 recipe/checkpoint |

我们保留负结果，避免只展示成功样本或把 kernel microbenchmark 写成完整模型加速。

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
- 不能将本仓库数字当作 RTX 5090、DGX Spark、A100、H100 或 8×GB200 结果；
- Turbo 是 practical approximation；BF16 baseline 才属于 fidelity 分母；
- 实际耗时会受 prompt 长度、驱动、温度、存储、host RAM 和后台负载影响。

如果你使用 Apple Silicon，请参见姐妹项目 [`Argus-AiTeam/minimax-h3-mac`](https://github.com/Argus-AiTeam/minimax-h3-mac)。两个仓库共享“真实设备、真实媒体、证据优先”的方法，但代码路径与性能数据完全独立。

---

## 项目与上游

- Argus Agent：<https://github.com/lbx154/Argus>
- 本项目：<https://github.com/Argus-AiTeam/minimax-h3-desktop>
- Mac 姐妹项目：<https://github.com/Argus-AiTeam/minimax-h3-mac>
- MiniMax-H3：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- vLLM-Omni：<https://github.com/vllm-project/vllm-omni>
- Turbo LoRA：<https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>
- NVLabs/Sana Sol-Engine：<https://github.com/NVlabs/Sana>

## 致谢

- **Argus**：自主完成本仓库的大部分 A6000 适配、实验执行、持续诊断、性能验证和文档工作；
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
