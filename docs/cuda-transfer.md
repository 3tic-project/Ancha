# CUDA 开发迁移包

复制整个迁移目录，包含 `ancha-source.tar.gz`、`ancha-assets.tar.gz`、`MANIFEST.json`、
`SHA256SUMS` 与本说明。两个压缩包都以 `Ancha/` 为顶层，解压到同一个新目录。

## 包含内容

- 源码包：完整源码、Cargo.lock、文档、脚本、合成测试及 `.git` 历史；固定 MSST / HyperACE / UVR
  参考实现、lightweight-separation-lab、算子实验源码、CUDA 回迁报告与实验脚本也放在忽略的 `NO_TRACK` 中。
- 资产包：六个已转换模型包、四个经典 MDX ONNX、原始 checkpoint 与配置；测试歌曲、固定
  3 秒和 30 秒 float32 WAV。无需在 CUDA 机器重新下载权重或转换。
- 排除：`target`、Python 环境、autotune / PTX 缓存、macOS 可执行文件、生成的分离音频、下载缓存与
  `__pycache__`。新 GPU 首次运行会重新编译内核并调优。

`NO_TRACK` / `NOTRACK` 继续被 Git 忽略；参考代码、模型和歌曲没有加入 Git。Git hooks 和本机 GUI 配置不迁移。
权重与测试歌曲用于本次开发交接，不随源码公开发布。

## Linux 解压与检查

```bash
# 在复制来的迁移目录中执行；四项都应显示 OK。
sha256sum -c SHA256SUMS
mkdir -p ~/work/ancha-cuda
tar -xzf ancha-source.tar.gz -C ~/work/ancha-cuda
tar -xzf ancha-assets.tar.gz -C ~/work/ancha-cuda
cd ~/work/ancha-cuda/Ancha
git config core.ignorecase false
git status --short                         # 应为空
git fsck --full
git check-ignore NO_TRACK/models/deux/model.safetensors
```

`MANIFEST.json` 记录源码提交、每个文件的大小 / SHA-256 / 执行权限和两个压缩包的摘要。
还可用系统 Python 3.11 或更新版本逐个核对压缩包内部文件（无需安装依赖）：

```bash
python3 scripts/package-cuda-transfer.py --verify /path/to/迁移目录
```

开发机需要 Linux 编译工具（C/C++ linker、pkg-config）、Git、Rust 1.92.0，以及 NVIDIA 驱动与 CUDA Toolkit 的
NVRTC。项目的 `rust-toolchain.toml` 固定 Rust 版本。首次 Cargo 构建需要联网获取依赖，本包没有 vendor Cargo registry。
可选的独立参考验证需要 uv / Python 3.11；ffmpeg 仅在重新制作片段时需要。

## 构建、验证与开发

```bash
nvidia-smi
export PATH=/usr/local/cuda/bin:$PATH
# 若 NVRTC 动态库不在系统加载路径，补上实际 Toolkit 的 lib64：
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# 没有 nvcc 时，显式指定与已安装 Toolkit 匹配的版本；例如 CUDA 12.2：
# export CUDARC_CUDA_VERSION=12020

bash scripts/cuda-dev.sh build              # release CLI + 两个 SDK 示例；CUDA doctor
bash scripts/cuda-dev.sh test               # CPU 合成测试 + CUDA 契约 + 独立进程的分配失败测试
bash scripts/cuda-dev.sh reference          # 重建 PyTorch 2.2.2+cpu / ORT 参考环境
bash scripts/cuda-dev.sh parity             # 原有八模型 + 两个 Derur 模型；缺项或失败会退出非零
bash scripts/cuda-dev.sh smoke              # 十模型串行分离歌曲第 30 秒起的 30 秒，原生上下文
```

Linux 构建不使用 macOS `accelerate` feature。`build` 自动追加 CUDA 与 checkpoint 转换工具；推理本身不依赖
PyTorch、ONNX Runtime 或 Python。`test` 的最后一个测试会主动申请超过整块显存的缓冲，验证错误能被报告，
应在空闲 GPU 上执行。所有 CUDA 测试都需要真实 NVIDIA GPU。

新加入的 Mel Karaoke / MDX23C 已在本地 CPU / RX 580 WGPU 验证；**本次交接前尚未在 NVIDIA 执行这两个模型**。
特别需要验收 MDX23C 的 CUDA TDF GEMM / 转置卷积路径，先通过 `test` 与 `parity` 再优化。
原有八模型的 Tesla P4 CUDA 验证见 [CUDA 记录](cuda.md)，不能代替新模型验证。
Mel 的快速 parity 使用 3 秒单块自定义上下文；MDX23C 使用原生重叠。`smoke` 的 30 秒运行是完整功能检查，
并不是独立 PyTorch 的整段质量验收。比对要求 max_abs < 1e-3 且 waveform SNR > 50 dB。

针对新模型进行串行预热测速与波形消融，使用 `scripts/benchmark-derur.py`；全后端测速使用
`scripts/benchmark-backends.sh`。保持同一二进制、PCM、原生 chunk / overlap，不把首次调优计入热运行速度。
接口和已有实测见 [Derur 适配说明](derur-adapters.md)。开发结果和完整日志继续保存在 `NO_TRACK/runs`。

## 在本机重新打包

```bash
# 提交源码后执行；拒绝覆盖已有迁移目录，完成后自动校验每个内部文件。
python3 scripts/package-cuda-transfer.py --output NO_TRACK/transfers/ancha-cuda-new
```

脚本采用显式资产范围和参考源码范围，不会把输出迁移包再次打入包中；压缩使用 gzip level 1，
避免高压缩级别在大权重上耗费过多时间。原始模型、歌曲和历史运行记录仍保留在工作区。
