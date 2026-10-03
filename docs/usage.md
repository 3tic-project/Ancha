# 使用说明

Ancha 把一首歌分成若干条 float32 WAV。运行时只需要编好的 `ancha` 和 `models/` 里的模型。

| 位置 | 内容 |
|---|---|
| `models/` | 转换好的模型包、四个 ONNX，以及下载下来的原始 checkpoint |
| `audio/` | 用户自行准备的歌曲；命令以 `ReoNa - Amore.mp3` 为例，文件不随源码分发 |
| `outputs/` | 分离结果。每个任务一个子目录，里面是 WAV 和 `run.json` |

## 环境

- Rust 1.92.0（`rust-toolchain.toml` 会选定这个版本）
- 能解码 WAV、FLAC、MP3
- 用 NVIDIA GPU 时，需要驱动和 CUDA Toolkit 12.x

CPU 始终可用。显卡后端要在编译时打开，并在命令里写明 `--backend`。

## 构建

```bash
cargo build --release --locked

# macOS
cargo build --release --locked --features convert,accelerate

# NVIDIA。没有 nvcc 时设置 CUDARC_CUDA_VERSION，CUDA 12.2 写 12020。
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --locked --features cuda
```

确认后端能做一次小运算：

```bash
target/release/ancha doctor --backend cpu
target/release/ancha doctor --backend wgpu --device 0
target/release/ancha doctor --backend cuda --device 0
```

`inspect <模型包或 .onnx>` 只打印配置，不处理音频。

## 分离

```bash
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/leap-xe-voc \
  --backend cuda \
  --output outputs/leap-xe-voc
```

- `--backend` 取 `cpu`、`ndarray`、`wgpu`、`cuda`。日常 CPU 用 `cpu`。`ndarray` 是旧的 CPU 实现，用来对照。
- `--device N` 选择第几块 GPU，从 0 起。
- `--start`、`--duration` 按秒截取。省略 `--duration` 就处理整首，默认最长 1 小时，可用 `--max-seconds` 调整。
- `-o` 与 `--output` 相同。目录里已有结果时，这次成功后会换成新文件。
- 单声道会复制成双声道。多于两个声道会报错。采样率不是 44100 时会重采样到 44100。
- 中途失败或 Ctrl+C 时，原来的结果还在。

默认使用模型自己的块长和重叠，`run.json` 里记为 `native-context`。短片段仍按这个块长补齐，所以 3 秒不一定比 30 秒快一个数量级。

第一次在某块 GPU 上跑某个模型时，会编译并试选内核，可能多花几分钟。之后会用缓存。

各模型的命令见 [README 的快速开始](../README.md#快速开始)。十个模型串行处理同一首歌：

```bash
cargo run --release --features cuda --example separate_all -- --backend cuda
```

汇总写到 `outputs/separate-all-<后端>-<时间>/summary.json`，每个模型一个子目录。`--only deux,mdx23c-inst-voc-hq2` 可以只跑其中几个。单个模型的 SDK 例子是 `examples/separate`。

## 输出文件

| 模型 | 直接预测 | 另一条 |
|---|---|---|
| `leap-xe-voc` | `vocals.wav` | 残差 `instrument.wav` |
| `deux` | `vocals.wav` 与 `instrument.wav` | 两轨都是预测 |
| `hyperace-v2-voc` | `vocals.wav` | 残差 `instrument.wav` |
| `hyperace-v2-inst` | `instrument.wav` | 残差 `vocals.wav` |
| `mel-karaoke-aufr33-viperx` | `vocals.wav` | 残差 `instrument.wav` |
| `mdx23c-inst-voc-hq2` | `vocals.wav` 与 `instrument.wav` | 两轨都是预测 |
| `UVR_MDXNET_9482.onnx` | `vocals.wav` | 残差 `instrument.wav` |
| `UVR_MDXNET_KARA.onnx` | `vocals.wav` | 残差 `instrument.wav` |
| `UVR_MDXNET_KARA_2.onnx` | `instrument.wav` | 残差 `vocals.wav` |
| `UVR-MDX-NET-Inst_HQ_2.onnx` | `instrument.wav` | 残差 `vocals.wav` |

文件名与任务无关：人声一侧一律是 `vocals.wav`，伴奏和卡拉 OK 伴唱一律是 `instrument.wav`。残差是原混音减去预测。KARA 2 预测的是伴唱，所以 `instrument.wav` 来自模型，`vocals.wav` 是残差。ONNX 按文件内容识别，改名也可以。

`run.json` 记录后端、实际配置、输入与权重摘要、各阶段耗时和 RTF（总时间除以音频时长）。

2026-10-03 在 Tesla P4 上整曲跑过 Mel Karaoke 和 MDX23C（277.57 秒）：Mel 145 块、模型 199.7 秒、整次 219.3 秒；MDX23C 383 次前向、模型 487.8 秒、整次 536.7 秒。第一次编译内核时会更久。

## 常用参数

不确定时用快速开始里的命令即可。

| 参数 | 作用 |
|---|---|
| `--chunk-samples`、`--overlap` | 改成自定义上下文，`run.json` 标成 `custom-context`。适合试跑 |
| `--mdx-denoise` | 仅经典 MDX。正反各算一遍再平均，时间大约加倍 |
| `--mdx-overlap` | 仅经典 MDX。改变块步进，范围 0 到 0.95 |
| `--mdx-batch-size` | 仅经典 MDX。一次前向的块数，1 到 4，更占显存 |
| `--conv-strategy` | `auto`、`gemm` 或 `backend`。GPU 上的 MDX 默认走 GEMM |
| `--mdx-no-optimize` | 经典 MDX 关闭图折叠；MDX23C 关闭 CUDA 上的 TDF 加速 |

其余参数见 `ancha separate --help`。数值含义见[开发文档](development.md)和[架构说明](architecture.md)。

## 准备模型

`models/` 里已经有转换好的包和四个 ONNX 时，直接 `--model`。换一台机器时用脚本下载，再用 `convert`（编译时要带 `--features convert`）：

```bash
bash scripts/download-leap.sh
target/release/ancha convert models/bs_leap_xe_voc.ckpt \
  --preset leap-xe-voc --output models/leap-xe-voc

bash scripts/download-models.sh
target/release/ancha convert models/hyperace-v2-voc.ckpt \
  --preset hyperace-v2-voc --output models/hyperace-v2-voc
target/release/ancha convert models/hyperace-v2-inst.ckpt \
  --preset hyperace-v2-inst --output models/hyperace-v2-inst

bash scripts/download-derur-models.sh
target/release/ancha convert \
  models/derur-download/mel_band_roformer_karaoke_aufr33_viperx_sdr_10/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt \
  --preset mel-karaoke-aufr33-viperx --output models/mel-karaoke-aufr33-viperx
target/release/ancha convert \
  models/derur-download/MDX23C-8KFFT-InstVoc_HQ_2/MDX23C-8KFFT-InstVoc_HQ_2.ckpt \
  --preset mdx23c-inst-voc-hq2 --output models/mdx23c-inst-voc-hq2
```

预设还有 `leap-xe-inst`。Deux 要一份导出的配置，步骤在[开发文档](development.md#转换-deux)。四个 ONNX 由 `download-models.sh` 直接放到 `models/`，不用转换。

模型文件的使用许可与代码许可分开，见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
