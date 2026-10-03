# Ancha

本地、离线的音频分离工具。用 Rust 编写，提供命令行和 Rust SDK。分离时不需要 Python 或 PyTorch。

支持 WAV、FLAC、MP3。输出为 44.1 kHz、float32 双声道 WAV，并附带一份 `run.json`。模型放在 `models/`，结果放在 `outputs/`，示例歌曲放在 `audio/`。

## 构建

需要 Rust 1.92.0。首次构建要联网拉取依赖。

```bash
cargo build --release --locked
```

NVIDIA 机器加上 CUDA，并让 `nvcc` 在 PATH 里：

```bash
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --locked --features cuda
```

macOS 可使用 `cargo build --release --locked --features convert,accelerate`。下面的例子用 `--backend cuda`；没有显卡时改成 `--backend cpu`。

## 快速开始

先把自己的歌曲放到 `audio/`，下面以 `audio/ReoNa - Amore.mp3` 为例。模型和歌曲由用户准备，不随源码分发。
每条命令把结果写到 `outputs/` 下对应的目录，再跑一次会换上新结果。只要试听 30 秒，在命令末尾加上 `--start 30 --duration 30`。

```bash
# Leap Xe：人声，残差为伴奏
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/leap-xe-voc --backend cuda --output outputs/leap-xe-voc

# Deux：人声与伴奏都由模型预测
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/deux --backend cuda --output outputs/deux

# HyperACE：人声，或伴奏
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/hyperace-v2-voc --backend cuda --output outputs/hyperace-voc
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/hyperace-v2-inst --backend cuda --output outputs/hyperace-inst

# Mel Karaoke：主唱，残差为卡拉 OK 混音
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/mel-karaoke-aufr33-viperx --backend cuda --output outputs/mel-karaoke

# MDX23C：人声与伴奏都由模型预测
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/mdx23c-inst-voc-hq2 --backend cuda --output outputs/mdx23c

# 经典 MDX（ONNX，不用转换）
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/UVR_MDXNET_9482.onnx --backend cuda --output outputs/mdx-9482
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/UVR_MDXNET_KARA.onnx --backend cuda --output outputs/mdx-kara
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/UVR_MDXNET_KARA_2.onnx --backend cuda --output outputs/mdx-kara-2
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/UVR-MDX-NET-Inst_HQ_2.onnx --backend cuda --output outputs/mdx-inst-hq-2
```

各模型写出的文件名、参数和准备权重的方法见[使用说明](docs/usage.md)。构建、测试和数值比对见[开发文档](docs/development.md)。

### 各模型的输出

| 命令里的模型 | 结果 |
|---|---|
| `leap-xe-voc` | `vocals.wav`，残差 `instrument.wav` |
| `deux` | `vocals.wav`、`instrument.wav`，两轨都是预测 |
| `hyperace-v2-voc` | `vocals.wav`，残差 `instrument.wav` |
| `hyperace-v2-inst` | `instrument.wav`，残差 `vocals.wav` |
| `mel-karaoke-aufr33-viperx` | `vocals.wav`，残差 `instrument.wav` |
| `mdx23c-inst-voc-hq2` | `vocals.wav`、`instrument.wav`，两轨都是预测 |
| `UVR_MDXNET_9482.onnx` | `vocals.wav`，残差 `instrument.wav` |
| `UVR_MDXNET_KARA.onnx` | `vocals.wav`，残差 `instrument.wav` |
| `UVR_MDXNET_KARA_2.onnx` | `instrument.wav`，残差 `vocals.wav` |
| `UVR-MDX-NET-Inst_HQ_2.onnx` | `instrument.wav`，残差 `vocals.wav` |

每个模型都只写出这两个文件名。主唱、全部人声写到 `vocals.wav`，伴奏和卡拉 OK 伴唱写到 `instrument.wav`。残差是原混音减去预测。KARA 2 的预测是伴唱，所以 `instrument.wav` 是模型输出，`vocals.wav` 是残差。

## 更多文档

- [使用说明](docs/usage.md)
- [开发文档](docs/development.md)
- [架构与数值约定](docs/architecture.md)
- [HyperACE 与经典 MDX](docs/adapters.md)
- [Mel Karaoke 与 MDX23C](docs/derur-adapters.md)
- [CUDA 记录](docs/cuda.md)

来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
