# HyperACE 与经典 MDX 适配

本轮把 `NO_TRACK/lightweight-separation-lab` 的离线图分析推进为原生 Rust 整模型推理。
执行器为 Burn 0.21.0 + CPU NdArray / WGPU；ONNX 由 onnx-rs 0.1.2 直接解析。
运行时不依赖 Python、ONNX Runtime 或 LibTorch。开发用的独立参考环境保持在 NO_TRACK。

## 使用

macOS 本机完整构建：

```bash
cargo build --release --locked --features convert,accelerate
bash scripts/download-models.sh

target/release/ancha convert models/hyperace-v2-voc.ckpt \
  --preset hyperace-v2-voc --output models/hyperace-v2-voc
target/release/ancha convert models/hyperace-v2-inst.ckpt \
  --preset hyperace-v2-inst --output models/hyperace-v2-inst

# ONNX 可直接使用，无须导出中间模型包。
target/release/ancha inspect models/UVR_MDXNET_KARA_2.onnx
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/UVR_MDXNET_KARA_2.onnx --backend cuda \
  --start 30 --duration 30 --output outputs/mdx-kara-2

# HyperACE 使用原生 960000-sample context / overlap=4。
target/release/ancha separate 'audio/ReoNa - Amore.mp3' \
  --model models/hyperace-v2-voc --backend cuda \
  --start 30 --duration 3 --output outputs/hyperace-voc
```

Linux / Windows 去掉 `accelerate`。CPU 可用 `--backend cpu`；所有后端都需显式选择。
脚本核对 SHA256，已有文件校验通过后复用；不会把权重加入 Git。

## HyperACE v2

专用 adapter 保留 12 层轴向 RoFormer、62 个连续频带、per-band MLP/GLU，以及完整 SegmModel：

- 深度可分离卷积 Backbone，C3k / C3k2 残差块和四个 encoder scale。
- HyperACE 多尺度融合、两个高阶超图分支、低阶分支与旁路。
- 学习的 global_proto、平均/最大池化 context、query projection、head 平均、节点轴 softmax，
  `A·x` 与 `Aᵀ·edge`。
- 带学习 gamma 的 decoder、仅频率方向 PixelShuffle、TFC/TDF 与 1025-bin 输出。

卷积按 NCHW、原 stride / padding / groups 执行。InstanceNorm 在 H×W 上使用当前输入的
biased variance、eps=1e-8 和 affine；不折叠为固定 BN。Resize 显式实现 half-pixel 坐标、
边界 clamp 和 `align_corners=false`，覆盖奇数时间/频率尺寸。所有参数必须被严格消费，
缺失、错形状、非有限值和额外张量都会报错。

v2 人声、伴奏 checkpoint 均可转换。与标准 BS 的差别还包括 `zero_dc=false`、960000 chunk、
overlap=4，不能把普通 BS 权重或设置套入该模型。3 秒单块 parity 使用显式短上下文；它验证
数值实现，不证明缩短上下文保持原生音质。具体实测和当前验收范围见 [性能报告](adapters-performance.md)。

## MDX 模型语义与 DSP

| 注册的 ONNX | 任务 | 直接预测 | 残差 | FFT / F / T | compensate |
|---|---|---|---|---|---:|
| UVR_MDXNET_9482 | 全部人声分离 | all_vocals | instrumental | 6144 / 2048 / 256 | 1.035 |
| UVR_MDXNET_KARA | 主唱分离 | lead_vocals | karaoke_mix | 6144 / 2048 / 256 | 1.035 |
| UVR_MDXNET_KARA_2 | 主唱分离 | karaoke_mix | lead_vocals | 5120 / 2048 / 256 | 1.065 |
| UVR-MDX-NET-Inst_HQ_2 | 全部人声分离 | instrumental | all_vocals | 6144 / 3072 / 256 | 1.033 |

文件名可更改，注册按完整 SHA256 识别。未注册摘要拒绝运行，防止猜测 FFT 或预测轨语义。
普通任务的人声包含和声。写出的文件一律是 `vocals.wav` 和 `instrument.wav`：上表里的人声轨（all_vocals、lead_vocals）对应 `vocals.wav`，伴奏和卡拉 OK 伴唱（instrumental、karaoke_mix）对应 `instrument.wav`。KARA 2 的预测是伴唱，因此 `instrument.wav` 是模型输出。

44100 Hz 双声道、hop=1024，周期 Hann、center=true、reflect STFT；打包顺序为
`[left.real, left.imag, right.real, right.imag]`。网络输入的前三个频点清零，超出 F 的高频在
iSTFT 前补零。固定 chunk 为 `1024×255=261120` samples；trim=n_fft/2，左右零填充、
默认 step=chunk−n_fft、逐块 Hann 窗及 divider、compensate、最终原始长度都遵循固定 UVR 源码。
没有对输入/输出做峰值归一化或 clipping。原混音减预测轨得到语义明确的 residual。

默认单遍推理。`--mdx-denoise` 显式执行 `(f(x)−f(−x))/2`，成本约翻倍；
`--mdx-overlap 0.5` 显式改变 chunk step，属于不同推理配置。
这些选项以及 batch、实际 forward 次数均进入 run.json。RoFormer 的 attention/context 参数
不接受用于 ONNX；MDX 固定 FFT 和 T，不能靠任意缩短 chunk 冒充等配置加速。

## 执行器与加速

支持本次图中的 Conv2d、ConvTranspose2d、MatMul、BatchNormalization(eval)、Relu、
Add、Mul、Transpose。要求 opset=13、一个 F32 输入/输出、明确的 NCHW 维度；未知算子、属性、
外部权重、非有限 initializer 或训练态 BN 都报错，不静默降级。

图在加载时编译成索引指令，保持只读权重，使用消费计数在最后一个使用点释放中间句柄。
默认预计算固定 BN 的 scale/shift；仅当 ConvTranspose 输出只有一个消费者、不是图输出、
且后接 eval BN 时，把 BN 折叠进克隆的 weight/bias。原共享 initializer 保留，不修改其他消费者。
剩余 TDF MatMul 后的 BN 按通道预计算 affine，不错误折叠成公共二维 MatMul 权重。

默认跳过 `start >= trim + input_length` 的尾块：这些块与截取输出没有交集，不改变 OLA。
网络仍完整计算所有有贡献的 chunk；没有静音近似、频带删减、量化或注意力替换。
`--mdx-no-optimize` 可同时关闭这些优化，供相同二进制消融。

WGPU / CUDA 默认 `--conv-strategy auto` 把非分组、无空洞的 Conv 改写为滑窗 patch gather（`unfold`，
通道×ky×kx 顺序对应 `[out,in,kh,kw]` 权重）加一次 autotune GEMM。乘加项与原卷积相同，
只是累加顺序不同；CubeCL 在 RX 580 这类无 cooperative-matrix 的 GPU 上只提供直接卷积。
CPU 保持 Flex 原生 im2col 卷积，实测比 patch gather 更快。`--conv-strategy gemm|backend`
可强制任一路径，实际策略写入 run.json 的 `conv_strategy`。同一参数也作用于 HyperACE SegmModel 的非分组卷积，
其 `auto` 只在 CUDA 上选 GEMM（Tesla P4 实测快约 10%，见 [CUDA 记录](cuda.md)）。

`--mdx-batch-size 2` 可把多个独立固定形状 chunk 放入一次 forward；默认 1，上限 4。
这不改变每个块的 padding / context / OLA，但增加显存用量。最后不足整批时使用实际 batch。
批量大小不是跨块隐藏状态缓存；经典 MDX 没有 attention 或 KV cache。

`--backend cpu` 使用 Burn Flex（SIMD、Rayon gemm 与 im2col 卷积）；旧 NdArray 通过
`--backend ndarray` 保留，可选 `cpu-opt` 开启其 SIMD 卷积与多线程。上一轮 NdArray CPU 基准在进程启动前
设置 `RAYON_NUM_THREADS=1` 或 `4` 并固定 `VECLIB_MAXIMUM_THREADS=1`；本轮 Flex 基准使用默认线程。
环境与 build_features 保存在报告中。

所有任务仍使用完整主机 PCM 和 OLA，受 `--max-seconds` 样本预算保护；尚未实现磁盘流式 OLA。
设备工作区释放句柄不等于实测显存峰值，本文不把估计内存写作 profiler 结果。

## 独立参考与复现

```bash
bash scripts/fetch-reference.sh
bash scripts/fetch-extra-reference.sh
uv pip install --python NO_TRACK/.venv-parity/bin/python -r scripts/requirements-mdx-parity.txt

NO_TRACK/.venv-parity/bin/python scripts/verify_mdx.py \
  --uvr-source NO_TRACK/reference/uvr --input NO_TRACK/runs/clip-3s.wav \
  --model NO_TRACK/models/UVR_MDXNET_KARA_2.onnx \
  --rust-output NO_TRACK/runs/mdx-kara-2-wgpu-3s \
  --report NO_TRACK/runs/parity-mdx-kara-2-wgpu.json

# 空闲机器上串行执行，基线/优化顺序交替；各 3 次。
bash scripts/benchmark-mdx.sh
```

参考脚本直接从固定 UVR 文件中抽取原始 initialize/demix/run_model 方法；
网络由 CPU ONNX Runtime 1.20.1 执行，创建 session 前关闭 telemetry，不加载 GUI。
标准门槛 max_abs<1e-3 且 waveform SNR>50dB；这衡量实现一致性，不是有真值的 SDR。

HyperACE parity 使用 `scripts/verify_parity.py --hyperace-source NO_TRACK/reference/hyperace/bs_roformer.py`，
并指定对应 checkpoint、package、同一 3 秒 WAV 和单块 Rust 输出。

原始资料：[HyperACE 固定源码](https://huggingface.co/pcunwa/BS-Roformer-HyperACE/blob/5b1f8283125d5e4a3614d0e3635a636e09c84059/v2_voc/bs_roformer.py)、
[UVR 固定推理](https://github.com/Anjok07/ultimatevocalremovergui/blob/5517e0cf0d1acd16a1618eeedec596957523f9e1/separate.py)、
[UVR STFT](https://github.com/Anjok07/ultimatevocalremovergui/blob/5517e0cf0d1acd16a1618eeedec596957523f9e1/lib_v5/tfc_tdf_v3.py)。
权重与代码许可分别记录在 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
