# 开发与复现

工具链固定为 Rust 1.92.0 / Burn 0.21.0，提交 Cargo.lock。默认纯 Rust CPU 后端；
macOS 用 `accelerate` feature 链接系统 Accelerate BLAS。`wgpu` 为可选 GPU feature。
`convert` 只增加 checkpoint 读取工具，发布分离程序可不启用。

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --features convert --locked -- -D warnings
cargo test --workspace --features convert --locked
cargo build --release --features convert,wgpu,accelerate,onnx,cpu-opt --locked # macOS
```

CI 使用原创合成音频与微型权重，不下载真实权重或商业歌曲。GPU 编译检查不等于 GPU 执行测试。
本机实际硬件执行与 30 秒音频结果记录在 `docs/performance.md` 和 `docs/reports`。

开发期独立 PyTorch 参考环境只存在于 NO_TRACK。

```bash
uv venv --python 3.11 NO_TRACK/.venv-parity
uv pip install --python NO_TRACK/.venv-parity/bin/python -r scripts/requirements-parity.txt
bash scripts/fetch-reference.sh

# 使用同一 float32 WAV；132300 samples = 3 秒 @44.1 kHz。
ffmpeg -ss 30 -i 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  -t 3 -ac 2 -ar 44100 -c:a pcm_f32le NO_TRACK/runs/clip-3s.wav
target/release/ancha separate NO_TRACK/runs/clip-3s.wav \
  --model NO_TRACK/models/leap-xe-voc --output NO_TRACK/runs/parity-cpu \
  --backend cpu --chunk-samples 132300 --overlap 1
NO_TRACK/.venv-parity/bin/python scripts/verify_parity.py \
  --reference NO_TRACK/reference --checkpoint NO_TRACK/models/bs_leap_xe_voc.ckpt \
  --package NO_TRACK/models/leap-xe-voc --input NO_TRACK/runs/clip-3s.wav \
  --rust-output NO_TRACK/runs/parity-cpu --report NO_TRACK/runs/parity-cpu.json
```

单 chunk FP32 波形门槛为 max_abs < 1e-3 且 waveform SNR > 50 dB。这是与固定 Python
forward 的一致性门槛，不是干净源 SDR，也不能替代整轨边界和数据集质量回归。
PyTorch 2.2.2 是 Intel macOS 的实际验证版本；其它系统可另建环境，但报告必须记录版本。

Deux 使用精确导出的 librosa 二值 Mel 索引：

```bash
NO_TRACK/.venv-parity/bin/python scripts/make-deux-config.py configs/deux.json
target/release/ancha convert NO_TRACK/models/becruily_deux.ckpt \
  --preset deux --config configs/deux.json --output NO_TRACK/models/deux
```

自定义 config 只能表达 schema 1 已实现的结构。HyperACE v2 需显式 `family=hyperace-v2`，
维度 256、62 bands、完整空间分支及对应权重；推荐使用专用 preset。
转换工具对所有张量转 F32，校验 shape 与 dtype，并运行 CPU 架构构造检查。
本地模型包并不包含分发权重的许可授权。

### 性能消融

`scripts/benchmark-separation.sh` 在同模型、同 PCM、同 chunk / overlap 下顺序比较批量
投影和 `--flatten-linear`。替代布局可能更慢，报告保留小于 1 的 speedup；不因此改用短块。
GPU 参数 tuning 可对比 `--query-tile 128 --group-tile 4` 与 `512 / 16`，然后使用
`scripts/compare_runs.py` 检查两轨波形与运行契约。大的 tile 增加 scores 工作内存；
本机原生 Leap 块在 512 / 16 下约 430.5 MiB，默认 128 / 4 约 26.9 MiB。
这些是张量尺寸推导值，不是 profiler 测得的总显存峰值。

工作区无远端配置。提交前执行 `git check-ignore NO_TRACK/...` 和
`git ls-files`，确认原始音频、参考工程、完整日志与权重未进入索引。

## HyperACE / MDX 新流程

完整功能和 CPU 优化构建：

```bash
cargo build --release --locked --features convert,wgpu,accelerate,onnx,cpu-opt
cargo clippy --workspace --all-targets --locked --features convert,wgpu,accelerate,onnx,cpu-opt -- -D warnings
cargo test --workspace --locked --features convert,wgpu,accelerate,onnx,cpu-opt
```

跨平台 CI 使用 `convert,onnx,cpu-opt`；macOS 才添加 accelerate。
28 个合成测试新增空间 InstanceNorm、half-pixel resize、频率 shuffle、HyperACE preset、
ONNX 调度/BN folding/拒绝规则、batch 轴和 MDX DSP / 取消保护，不读取真实权重。
独立真实模型验证、下载、MDX 同二进制消融和批量设置见 [适配文档](adapters.md)。
CPU 可在启动前设置 `RAYON_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1`，避免卷积线程与 BLAS
线程叠加过多；测量时同时记录 feature、环境和实际耗时，不把线程设置值当作监测到的线程数。
参考环境安装 `scripts/requirements-mdx-parity.txt`，Rust 发布运行时不依赖它。
