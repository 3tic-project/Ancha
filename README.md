# Ancha

本地、离线的 Rust RoFormer / MDX 音频分离项目。提供 CLI 和 Rust SDK，运行时无需 Python、
PyTorch 或 LibTorch。原有 NO_TRACK 设计与算子实验已整理为可构建、可测试的工程。

支持 WAV / FLAC / MP3 输入、片段截取、sinc 重采样、原生全局注意力、重叠合成、
float32 WAV 输出和可追溯的 run.json。模型与素材均保留在 NO_TRACK，Git 不跟踪。

| 模型 | 当前状态 |
|---|---|
| Leap Xe vocals | 真实权重已加载；CPU / RX 580 WGPU 分离与独立 PyTorch FP32 波形比对通过 |
| Leap Xe instrumental | 同架构配置与转换预设已实现；该 checkpoint 尚未单独验收 |
| Deux | Mel 索引固定导出、双原生输出头已实现；CPU / WGPU 分离及两个输出的 PyTorch 波形比对通过 |
| HyperACE v2 vocals / instrumental | 完整 SegmModel 已实现；两个 checkpoint 的 WGPU 波形比对通过，vocals 另通过 CPU 比对 |
| 经典 MDX ONNX | 9482、KARA、KARA 2、Inst HQ 2 原生 Rust 推理；四模型 WGPU 对齐 UVR，KARA 2 / HQ 2 另通过 CPU 比对 |

以上 8 个模型（9 个输出）另在 NVIDIA Tesla P4 的 CUDA 后端与 Linux Xeon CPU 上全部通过同一组参考比对。
同一 NVIDIA GPU 上的 WGPU（Vulkan）对 Deux 与 HyperACE inst 比对失败（与 CUDA 之前的版本相同），NVIDIA 显卡请用 CUDA。

HyperACE 与经典 MDX 的适配、使用和任务语义见 [新增适配文档](docs/adapters.md)。
CPU / WGPU 推理速度审计、算子折叠与当前实测见 [速度优化记录](docs/speed-optimization.md)；
CUDA 后端的适配、优化、profile 与 Linux 三后端实测见 [CUDA 记录](docs/cuda.md)；
上一轮速度见 [适配性能记录](docs/adapters-performance.md)。目前仍未完成 Leap inst 独立验收，
CUDA 的 GEMM 仍为 CubeCL 通用内核（未接 cuBLAS / Tensor Core）。
早期 Leap / Deux 的测试和优化保留在 [性能记录](docs/performance.md)。

## 构建与本机使用

Rust 1.92.0，首次构建需联网获取 Cargo.lock 中的依赖。默认 feature 已包含 CPU 优化（`cpu-opt`）、WGPU 与
MDX ONNX；`convert`（checkpoint 转换）、macOS `accelerate` 与 NVIDIA `cuda` 按需追加。macOS 本机推荐：

```bash
cargo build --release --locked --features convert,accelerate
target/release/ancha doctor --backend wgpu
target/release/ancha inspect NO_TRACK/models/leap-xe-voc

# 在歌曲第 30 秒开始，分离 30 秒；使用模型原生 chunk / overlap。
target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/leap-xe-voc --backend wgpu \
  --start 30 --duration 30 --output NO_TRACK/runs/my-30s
```

输出目录必须尚不存在。完成后包含 `vocals.wav`、`instrumental.wav`、`run.json`。
默认 FP32、44.1 kHz 双声道；输入单声道会复制成双声道。单头模型的另一轨标为 residual，
Deux 两轨都标为 predicted。运行报告记录有效参数、PCM/权重摘要、各阶段耗时和 RTF。

Linux / Windows 不启用 Apple Accelerate：

```bash
cargo build --release --locked            # CPU + WGPU + MDX ONNX
target/release/ancha doctor --backend cpu

# NVIDIA GPU 只需再加 cuda：需要驱动与 CUDA Toolkit 12.x 的 NVRTC；构建时 nvcc 在 PATH 中（或设
# CUDARC_CUDA_VERSION=12020 一类的值与驱动匹配）。运行时动态加载，不链接 CUDA 库。
PATH=/usr/local/cuda/bin:$PATH cargo build --release --locked --features cuda
target/release/ancha doctor --backend cuda
```

同一个二进制用 `--backend cpu|ndarray|wgpu|cuda`（GPU 另可 `--device N`）在启动时切换后端。
`--no-default-features` 仍可构建只含 CPU 的最小版本。

`--duration` 只截取待处理片段；默认依然按原生块长补齐，因此短片段不必然按时长线性提速。
若希望快速技术测试，可显式改变上下文并保留对应报告：

```bash
target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/leap-xe-voc --backend wgpu \
  --start 30 --duration 3 --chunk-samples 132300 --overlap 1 \
  --output NO_TRACK/runs/my-3s
```

`--chunk-samples` / `--overlap` 会改变分离上下文，标记为 `custom-context`；它们不是等质量加速。
attention 分块默认按 `--max-score-mib`（512）自动选择，`--query-tile` / `--group-tile`
可显式指定；它们都只调节完整 K/V 注意力的内部计算分块。
`--linear-layout auto` 在 CUDA 上把投影的独立行合并为一次 GEMM，其它后端保留本机实测更快的
批量投影；`batched` / `flattened`（旧参数 `--flatten-linear`）可强制任一布局做消融。
CPU、WGPU、CUDA 必须显式选择；没有静默后端回退。Ctrl+C 可在模型层／chunk 边界取消。
`--backend cpu` 使用纯 Rust Burn Flex，`--backend ndarray` 保留旧 CPU 后端供对比。
WGPU / CUDA 启用 autotune：某模型与上下文首次运行会先实测内核（可达数分钟），结果缓存后复用；
CUDA 另按 GPU 架构与模型缓存 NVRTC 编译结果。GPU 设备线程上的显存分配失败会使任务报错，
不会发布结果。

## 示例

`examples/` 演示 SDK 调用方式，后端由 `ancha::backend` 在运行时选择：

```bash
# 单个模型：目录为 RoFormer 包（Leap / Deux / HyperACE），.onnx 为经典 MDX。
cargo run --release --example separate -- 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  NO_TRACK/models/deux NO_TRACK/runs/example-deux --backend wgpu --start 30 --duration 30

# 全部 8 个模型完整分离一首歌并记录性能（NVIDIA 加 --features cuda）。
cargo run --release --features cuda --example separate_all -- --backend cuda
```

`separate_all` 在一个进程里串行运行 Leap、Deux、HyperACE voc / inst 与四个 MDX，各自使用原生上下文和
后端默认设置。输出写到 `NO_TRACK/runs/examples/separate-all-<后端>-<时间>/<模型>/`，每个模型结束后更新
`summary.json`（主机与设备、各阶段耗时、首块 / 稳态每块耗时、RTF、各轨峰值与 RMS、CUDA 显存占用）；
`--only`、`--start`、`--duration` 可缩小范围。本机 CUDA 全曲结果见 [CUDA 记录](docs/cuda.md#整首歌全模型示例)。

## 新工作区准备模型

当前工作区的已转换模型包位于 `NO_TRACK/models/leap-xe-voc` 和 `NO_TRACK/models/deux`。
Git 仓库不含权重。另一个工作区可按固定修订下载与转换：

```bash
bash scripts/download-leap.sh
target/release/ancha convert NO_TRACK/models/bs_leap_xe_voc.ckpt \
  --preset leap-xe-voc --output NO_TRACK/models/leap-xe-voc
```

下载脚本核验发布方 LFS SHA256；转换全程 Rust，输出 F32 Safetensors、摘要与 manifest，
并核对每个张量键和形状。Deux 配置／参考环境步骤见 [开发文档](docs/development.md)。
代码与权重许可分开记录，来源见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

## 测试与结构

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --features convert --locked -- -D warnings
cargo test --workspace --features convert --locked
target/release/ancha bench --tokens 256 --iterations 5

# 在空闲机器上顺序比较同模型、同 PCM、同上下文的两种投影布局。
bash scripts/benchmark-separation.sh
```

经典 MDX ONNX 可直接作为 `--model` 输入，无须转换：

```bash
target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/UVR_MDXNET_KARA_2.onnx --backend wgpu \
  --start 30 --duration 30 --output NO_TRACK/runs/my-karaoke
```

KARA 2 输出 `karaoke_mix.wav` 和残差 `lead_vocals.wav`，不会误标为全部人声。

核心为三个职责独立的 crate：`ancha-audio`、`ancha-models`、`ancha-kernels`；
根包负责 SDK、CLI、任务与报告。CI 不依赖真实模型或歌曲，端到端测试在临时目录内生成
原创微型权重和音频。模块边界、DSP 契约、主机内存限制与未实现项见 [架构文档](docs/architecture.md)。

根目录 Git 已初始化为 main，NO_TRACK / NOTRACK、target、完整音频、权重和本机临时输出
均在 .gitignore 中排除。脱敏的数值／速度报告位于 `docs/reports`。
