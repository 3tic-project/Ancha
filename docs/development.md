# 开发

可运行的模型在 `models/`，示例歌曲在 `audio/`，分离结果在 `outputs/`。参考实现、Python 环境和实验记录仍在 `NO_TRACK/`。

日常使用见[使用说明](usage.md)。本文是构建、测试、比对和换机。实现约定在[架构说明](architecture.md)。各模型的适配记录：

- [HyperACE 与经典 MDX](adapters.md)
- [Mel Karaoke 与 MDX23C](derur-adapters.md)
- [CUDA 记录](cuda.md)、[换机迁移](cuda-transfer.md)
- 速度与历史验收：[速度优化](speed-optimization.md)、[适配性能](adapters-performance.md)、[早期性能记录](performance.md)
- 数字原文在 `docs/reports/`

工具链固定为 Rust 1.92.0 / Burn 0.21.0，提交 Cargo.lock。默认 CPU 后端为纯 Rust 的 Burn Flex；
`--backend ndarray` 保留旧 NdArray，`cpu-opt` 与 macOS `accelerate`（系统 BLAS）只作用于它。
默认 feature 为 `wgpu`、`onnx`、`cpu-opt`；`wgpu` 启用 Burn fusion 与 autotune。`cuda` 同样启用 fusion 与 autotune，
只需在默认之上追加，与 CPU / WGPU 共存于同一二进制，运行时用 `--backend` 选择；它
经 CubeCL 用 NVRTC 在运行时编译内核，并动态加载 libcuda / libnvrtc；构建时 cudarc 从 PATH 中的
`nvcc --version` 确定绑定的 CUDA 版本，没有 nvcc 时用 `CUDARC_CUDA_VERSION`（如 12.2 写 `12020`）指定，
否则会回落到最新版本绑定，可能与旧驱动不匹配。`convert` 只增加 checkpoint 读取工具，
发布分离程序可不启用。

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --features convert --locked -- -D warnings
cargo test --workspace --features convert --locked
cargo build --release --features convert,accelerate --locked # macOS
PATH=/usr/local/cuda/bin:$PATH cargo build --release --features cuda --locked # Linux + NVIDIA
# 仅在有 NVIDIA GPU 的空闲机器上。分配失败测试会占满显存，单独进程运行。
cargo test --release --locked --features cuda,convert \
  --test cuda_contracts --test cuda_mdx23c_contracts -- --test-threads=1
cargo test --release --locked --features cuda,convert \
  --test cuda_device_failure -- --test-threads=1
# 或者：bash scripts/cuda-dev.sh build|test|reference|parity|smoke
# 手写内核的算子级正确性与速度（对照 Burn 路径）。
cargo run --release -p ancha-models --features cuda --example fused_attention -- 90 1722
cargo run --release -p ancha-models --features cuda --example fused_linear -- 154980
```

CI 使用原创合成音频与微型权重，不下载真实权重或商业歌曲。GPU 编译检查不等于 GPU 执行测试；
CI 对 `cuda` 只做 `cargo check`（固定 `CUDARC_CUDA_VERSION=12020`）。
硬件上的速度和比对数字在 [CUDA 记录](cuda.md)、[Derur 适配说明](derur-adapters.md) 和 `docs/reports/`。早期 CPU / WGPU 记录仍在 [性能记录](performance.md)。

WGPU autotune 在每个新的算子形状首次出现时实测候选内核，在仓库内运行时缓存于
`target/autotune`，仓库外运行时位于系统用户缓存目录的 `cubecl`。首次使用某模型与上下文可能
多花数十秒到数分钟，`timings.model_call_seconds[0]` 会体现；测速须先预热并分开报告冷/热耗时。
CUDA 使用同一 autotune 缓存，另把 NVRTC 生成的 PTX 存入同一缓存根下的
`ptx-sm<算力>/<模型文件名>/`；未设置时每个进程都要重新编译全部内核（Tesla P4 上 MDX 约 12 秒、
RoFormer 短块约 25 秒）。更换 GPU 架构时自动使用新目录；降级驱动或 CubeCL 版本变化后可删除该目录。
在 cubecl.toml / Burn.toml 的 `[compilation] cache` 中可显式指定位置，此时不再按模型分区。

开发期独立 PyTorch 参考环境只存在于 NO_TRACK。

```bash
uv venv --python 3.11 NO_TRACK/.venv-parity
uv pip install --python NO_TRACK/.venv-parity/bin/python -r scripts/requirements-parity.txt
bash scripts/fetch-reference.sh

# 使用同一 float32 WAV；132300 samples = 3 秒 @44.1 kHz。
mkdir -p outputs
ffmpeg -ss 30 -i 'audio/ReoNa - Amore.mp3' \
  -t 3 -ac 2 -ar 44100 -c:a pcm_f32le outputs/clip-3s.wav
target/release/ancha separate outputs/clip-3s.wav \
  --model models/leap-xe-voc --output outputs/parity-cpu \
  --backend cpu --chunk-samples 132300 --overlap 1
NO_TRACK/.venv-parity/bin/python scripts/verify_parity.py \
  --reference NO_TRACK/reference --checkpoint models/bs_leap_xe_voc.ckpt \
  --package models/leap-xe-voc --input outputs/clip-3s.wav \
  --rust-output outputs/parity-cpu --report outputs/parity-cpu.json
```

单 chunk FP32 波形门槛为 max_abs < 1e-3 且 waveform SNR > 50 dB。这是与固定 Python
forward 的一致性门槛，不是干净源 SDR，也不能替代整轨边界和数据集质量回归。
PyTorch 2.2.2 是 Intel macOS 的实际验证版本；Linux 上使用同版本的 CPU wheel（`torch==2.2.2+cpu`，
来自 PyTorch CPU 索引或其镜像），其余依赖按 requirements 安装；报告会记录实际版本。

原有 8 个模型可用一条命令按后端复核，结果汇总到 `summary.json`：

```bash
ANCHA_BACKENDS="cuda wgpu cpu" bash scripts/parity-matrix.sh
```

比对脚本默认使用 `models/`，也兼容旧迁移包的 `NO_TRACK/models/`；可用 `ANCHA_MODELS` 指定。
固定片段优先使用 `NO_TRACK/runs/clip-3s.wav`，不存在时使用上述 `outputs/clip-3s.wav`；可用 `ANCHA_CLIP_3S` 指定。

Mel Karaoke / MDX23C 使用 `scripts/parity-derur.sh`，不放进上面的八模型矩阵。下载、参考环境、合成测试和速度消融见 [Derur 适配文档](derur-adapters.md)。2026-10-03 已在 Tesla P4 上跑过这两个模型的 CUDA 比对，以及 HyperACE 两个头（确认共享的 InstanceNorm）。整曲命令和一次计时见[使用说明](usage.md#整曲示例)。

### 转换 Deux

Deux 使用精确导出的 librosa 二值 Mel 索引：

```bash
NO_TRACK/.venv-parity/bin/python scripts/make-deux-config.py configs/deux.json
target/release/ancha convert models/becruily_deux.ckpt \
  --preset deux --config configs/deux.json --output models/deux
```

自定义 config 只能表达 schema 1 已实现的结构。HyperACE v2 需显式 `family=hyperace-v2`，
维度 256、62 bands、完整空间分支及对应权重；推荐使用专用 preset。
转换工具对所有张量转 F32，校验 shape 与 dtype，并运行 CPU 架构构造检查。
本地模型包并不包含分发权重的许可授权。

### 性能消融

`scripts/benchmark-separation.sh` 在同模型、同 PCM、同 chunk / overlap 下顺序比较批量
投影和 `--flatten-linear`。替代布局可能更慢，报告保留小于 1 的 speedup；不因此改用短块。
默认 attention 分块由 `--max-score-mib` 自动确定；显式 `--query-tile 128 --group-tile 4`
可复现上一轮默认分块。CPU 的 `--host-threads 1` 关闭分组并行，MDX 的
`--conv-strategy gemm|backend` 比较两种卷积路径；然后用 `scripts/compare_runs.py` 检查波形。
大的 tile 增加 scores 工作内存；本机原生 Leap 块在 512 / 16 下约 430.5 MiB，128 / 4 约 26.9 MiB。
这些是张量尺寸推导值，不是 profiler 测得的总显存峰值。

提交前执行 `git check-ignore NO_TRACK/...` 和
`git ls-files`，确认原始音频、参考工程、完整日志与权重未进入索引。

### 2026-10-03 CUDA 回迁与 P4 验收

交接时 `main` 从 `f55bfb1` 收到 `d8ac0d2` 及后续提交。补丁、参考比对和实验脚本备份在
`NO_TRACK/imports/cuda-transfer-20261003`。Git 继续排除 `NO_TRACK` / `NOTRACK`。

macOS Intel 上的编译、Clippy 和若干 CPU / WGPU 比对见
[cuda-integration-macos.json](reports/cuda-integration-macos.json)。
NVIDIA 硬件契约、Mel / MDX23C 的 3 秒比对、HyperACE 复测，以及 Amore 整曲，是在 Tesla P4 上做的。
数字分别在 [CUDA 记录](cuda.md)、[Derur 适配说明](derur-adapters.md) 和
[derur-verification.json](reports/derur-verification.json)。早期八模型整曲仍见
[cuda-kernels.json](reports/cuda-kernels.json)。

Linux CUDA 机器的固定入口：

```bash
bash scripts/cuda-dev.sh build       # release CLI、示例、doctor
bash scripts/cuda-dev.sh test        # 格式、CPU 合成测试、CUDA 契约、显存失败
bash scripts/cuda-dev.sh reference   # 重建 PyTorch 2.2.2 CPU 参考环境（需要 uv）
bash scripts/cuda-dev.sh parity      # 原八模型 + Mel / MDX23C
bash scripts/cuda-dev.sh smoke       # 十模型各跑歌曲中段 30 秒
```

打包与解压步骤在[迁移说明](cuda-transfer.md)。`scripts/package-cuda-transfer.py` 生成源码包和资产包并逐文件校验。

## HyperACE / MDX 新流程

完整功能和 CPU 优化构建：

```bash
cargo build --release --locked --features convert,accelerate
cargo clippy --workspace --all-targets --locked --features convert,accelerate -- -D warnings
cargo test --workspace --locked --features convert,accelerate
```

跨平台 CI 使用默认 feature 加 `convert`；macOS 才添加 accelerate。
31 个合成测试覆盖空间 InstanceNorm、half-pixel resize、频率 shuffle、HyperACE preset、
ONNX 调度/BN folding/拒绝规则、GEMM 卷积与后端卷积一致、batch 轴和 MDX DSP / 取消保护；
微型 RoFormer 的非相邻同宽频带、非单位 gamma，在 NdArray 与 Flex、手动与自动分块、
单线程与分组并行之间保持波形一致；新增权重摘要不匹配与设备线程故障的拒绝测试。测试不读取真实权重。
独立真实模型验证、下载、MDX 同二进制消融和批量设置见 [适配文档](adapters.md)。
Flex CPU 的卷积/矩阵乘走 Rayon，可在启动前设置 `RAYON_NUM_THREADS`；RoFormer 另有
`--host-threads` 控制分组 worker。旧 NdArray 测量建议同时设 `VECLIB_MAXIMUM_THREADS=1`，
避免卷积线程与 BLAS 线程叠加；测量时记录 feature、环境和实际耗时。
参考环境安装 `scripts/requirements-mdx-parity.txt`，Rust 发布运行时不依赖它。

## 迁移到其它机器

完整打包、校验、Linux CUDA 构建和十模型验收入口见 [CUDA 开发迁移包](cuda-transfer.md)。
`scripts/package-cuda-transfer.py` 生成源码 / 资产两包并逐文件验证，`scripts/cuda-dev.sh` 提供构建与测试步骤。

迁移包分为源码与资产两部分。源码包含 `.git`、文档、脚本和 NO_TRACK 中的参考代码与实验记录；
打包脚本保留原有资产布局：`NO_TRACK/models/`（checkpoint、已转换模型包、四个 ONNX）、`NO_TRACK/test_file/`
以及 `NO_TRACK/runs/` 中的固定 3 秒和 30 秒 float32 WAV。日常 CLI 的 `models/` / `audio/` 可作为它们的本地别名。不迁移 `target`、
`NO_TRACK/.venv-parity`（在新机器按 requirements 重建）、macOS 二进制、本机运行产物和 autotune
缓存；autotune 结果与 GPU、驱动和 CubeCL 版本绑定，新机器首次运行会重新调优。

Linux 构建去掉 `accelerate`：`cargo build --release --locked`（需要转换时加 `--features convert`），
有 NVIDIA GPU 时再加 `cuda`。后端为 CPU（Flex / NdArray）、WGPU（Linux 上走 Vulkan）与 CUDA。
迁移后先用 `scripts/parity-matrix.sh` 按后端复核数值，再用 `scripts/benchmark-backends.sh`
（同一二进制、多后端串行，先预热缓存）和 `scripts/benchmark-matrix.sh`（新旧二进制对比）测速，
与 [速度优化记录](speed-optimization.md)、[CUDA 记录](cuda.md) 的 profile 记录对照；
对照基线可从提交 `4221ed0`（上一轮）或 `f55bfb1`（CUDA 之前）构建。
