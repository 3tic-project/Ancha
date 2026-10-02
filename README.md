# Ancha

本地、离线的 Rust RoFormer 音频分离项目。提供 CLI 和 Rust SDK，运行时无需 Python、
PyTorch 或 LibTorch。原有 NO_TRACK 设计与算子实验已整理为可构建、可测试的工程。

支持 WAV / FLAC / MP3 输入、片段截取、sinc 重采样、原生全局注意力、重叠合成、
float32 WAV 输出和可追溯的 run.json。模型与素材均保留在 NO_TRACK，Git 不跟踪。

| 模型 | 当前状态 |
|---|---|
| Leap Xe vocals | 真实权重已加载；CPU / RX 580 WGPU 分离与独立 PyTorch FP32 波形比对通过 |
| Leap Xe instrumental | 同架构配置与转换预设已实现；该 checkpoint 尚未单独验收 |
| Deux | Mel 索引固定导出、双原生输出头已实现；CPU / WGPU 分离及两个输出的 PyTorch 波形比对通过 |
| HyperACE v2 | 专用 SegmModel 分支尚未实现；严格加载会报错 |

这是有实际分离能力的预览版，尚未完成 NO_TRACK PRD 中五个 checkpoint、CUDA 与三后端
全量验收。实际速度、GPU 验证与支持范围见 [性能记录](docs/performance.md)。

## 构建与本机使用

Rust 1.92.0，首次构建需联网获取 Cargo.lock 中的依赖。macOS 本机推荐：

```bash
cargo build --release --locked --features convert,wgpu,accelerate
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
cargo build --release --locked --features convert,wgpu
# 使用 CPU，可省略 wgpu feature。
target/release/ancha doctor --backend cpu
```

`--duration` 只截取待处理片段；默认依然按原生块长补齐，因此短片段不必然按时长线性提速。
若希望快速技术测试，可显式改变上下文并保留对应报告：

```bash
target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/leap-xe-voc --backend wgpu \
  --query-tile 512 --group-tile 16 \
  --start 30 --duration 3 --chunk-samples 132300 --overlap 1 \
  --output NO_TRACK/runs/my-3s
```

`--chunk-samples` / `--overlap` 会改变分离上下文，标记为 `custom-context`；它们不是等质量加速。
`--query-tile` / `--group-tile` 则只调节完整 K/V 注意力的内部计算分块。
默认保留本机实测更快的批量投影布局，`--flatten-linear` 提供实验性替代布局供消融。
CPU、WGPU 必须显式选择；没有静默后端回退。Ctrl+C 可在模型层／chunk 边界取消。

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

核心为三个职责独立的 crate：`ancha-audio`、`ancha-models`、`ancha-kernels`；
根包负责 SDK、CLI、任务与报告。CI 不依赖真实模型或歌曲，端到端测试在临时目录内生成
原创微型权重和音频。模块边界、DSP 契约、主机内存限制与未实现项见 [架构文档](docs/architecture.md)。

根目录 Git 已初始化为 main，NO_TRACK / NOTRACK、target、完整音频、权重和本机临时输出
均在 .gitignore 中排除。脱敏的数值／速度报告位于 `docs/reports`。
