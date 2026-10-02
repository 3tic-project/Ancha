# 本机测试与性能记录

2026-10-02，本机 Intel i5-12400（6 核 / 12 线程）、64 GB RAM、AMD Radeon RX 580 8 GB，
macOS 15.7.7 / x86_64，Rust 1.92.0、Burn 0.21.0、CubeCL 0.10.0。
release 二进制启用 `convert,wgpu,accelerate`；GPU 使用 WGPU 的 Metal adapter。
运行时不加载 Python / PyTorch / LibTorch。

## 真实音频与权重

测试素材为用户本地 `NO_TRACK/test_file/ReoNa - Amore.mp3`，44.1 kHz 双声道。
FFmpeg 在歌曲第 30 秒起截出 3 秒与 30 秒 float32 WAV，作为两种实现共用的精确 PCM。
片段、完整 checkpoint、已转换模型包、WAV 产物和完整日志都保留在 NO_TRACK。

| 项目 | 实际结果 |
|---|---|
| Leap Xe vocals | 66,845,708 参数，983 个 tensor 全部严格加载 |
| Deux | 1,188 个 tensor 全部严格加载，FP16 checkpoint 转为 F32 包，保留两个原生头 |
| Leap 30 秒原生分块 | 6 chunks，44,100 Hz / stereo，1,323,000 samples / channel |
| 30 秒完整墙钟时间 | 483.375 s，RTF 16.113；包含载入、解码、DSP、推理、OLA 与 WAV 写出 |
| 30 秒模型阶段 | 481.675 s |
| 30 秒残差重建 max_abs | 5.9604645e-8 |

30 秒 native-context 保留原生 chunk=881559 / overlap=2。首尾反射 border 也参与模型计算，
因此实际处理工作量超过 30 秒；不缩短模型上下文来冒充相同配置加速。
这次完整测试期间存在其它开发期 CPU 任务，保留真实墙钟结果，不能把它当作空闲系统的稳定吞吐。
该机器在这套全局 FP32 推理配置下没有达到实时分离。

权重下载已核对发布方固定修订的 LFS 摘要：

| Checkpoint | SHA256 |
|---|---|
| Leap Xe voc | `b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d` |
| Deux | `10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa` |

## 独立 FP32 波形一致性

使用固定 MSST `84b1eac0887756b4f1a9d7a1ff49105939749ed2` 原始 forward、
PyTorch 2.2.2 / NumPy 1.26.4、同一 132300-sample WAV。为隔离 forward，双方都执行单个
3 秒 chunk，不做跨块 OLA；Rust 的这一设置标为 custom-context。

| 模型 / 后端 | 输出 | max_abs | 波形 SNR |
|---|---|---:|---:|
| Leap / CPU | vocals | 5.7220e-6 | 108.12 dB |
| Leap / WGPU | vocals | 3.7551e-6 | 111.76 dB |
| Deux / CPU | vocals | 1.9744e-6 | 124.00 dB |
| Deux / CPU | instrumental | 1.7583e-6 | 123.63 dB |
| Deux / WGPU | vocals | 9.3877e-7 | 123.78 dB |
| Deux / WGPU | instrumental | 1.1921e-6 | 120.97 dB |

以上均通过 max_abs < 1e-3 且 waveform SNR > 50 dB 的门槛。
waveform SNR 是 Rust 与参考实现之间的数值一致性；没有干净源真值，因此没有 SDR、
人声提取质量分数或数据集平均质量结论。完整歌曲、Leap Xe inst、HyperACE 和 CUDA 尚未验收。

## 已采用的优化与消融

1. **矩阵乘法注意力**：生产实现使用 Burn GEMM，原 acceleration-lab 标量实现保留作独立参考。
   4 groups × 2 heads × 256 tokens × dim 64，5 次计时中位数：标量 30.802 ms、GEMM 9.599 ms，
   算子加速 3.21×，max_abs 1.56e-7。这个倍数只适用于合成 attention 算子。
2. **精确 query/group 分块**：原生 Leap 的 dense scores 一个张量约 7.95 GiB，
   默认 128 / 4 每次 scores 上界约 26.9 MiB。完整 K/V 与双向 attention 语义保留。
   这是张量尺寸推导的内存改善，不是 profiler 实测总显存峰值。
3. **DSP 复用**：FFT 计划、窗、频谱、时间与 scratch buffers 跨 chunk / channel 复用。
   没有每帧分配 FFT scratch，也没有每个 Transformer 层 CPU readback；模型时间包括最后的同步读回。
4. **片段解码**：达到请求时长后停止解码，避免为 3 秒测试读入整首 277 秒的歌曲。
5. **投影布局消融**：在相同 WGPU / PCM / chunk / overlap 下，批量投影 6.175 s，
   合并独立行的替代布局 12.702 s；输出逐采样相同。因此默认保留较快的批量布局，
   替代方案仅供 `--flatten-linear` 实验，不宣称它是提速。
6. **短块 tile tuning**：同一 3 秒上下文，128 / 4 总耗时 6.175 s、模型 5.327 s；
   512 / 16 总耗时 5.478 s、模型 4.488 s。模型阶段约 1.19×，总时间约 1.13×，两轨逐采样相同。
   为单次观察，后一测试存在同时进行的轻度 CPU 工作；需要在目标硬件重复测量。
   更大的 tile 在原生长块上表现较慢，已取消该长块试验，输出目录没有发布。
   默认仍是已完成 30 秒测试的 128 / 4；512 / 16 只建议用于本文的短片段实验。

两种上下文的时间不能横向解释为等质量速度比较。短块预览与 native-context 结果都记录实际参数。
CPU 阶段的初次 smoke 测试还包括 3 秒 Leap 90.243 s、Deux 51.449 s，存在并行开发负载，
只作为跑通证据，不用于和 PyTorch / GPU 报告整模型加速倍数。

## 测试、取消与报告

19 项 Rust 测试覆盖 DSP 边界、重采样脉冲位置、音频截取、全局注意力、epsilon 语义、
缓存准入、严格加载、两种模型的完整流水线、原生双头、残差、取消和输出目录保护。
微型权重与音频均在临时目录生成，不依赖 NO_TRACK。CPU、conversion、WGPU feature
均通过 fmt、Clippy -D warnings、test 与 release build。

原有融合 norm WGSL 在真实 RX 580 / Metal 上执行 5 组 case，max_abs <= 7.16e-7。
对运行中的真实长块试验发送 SIGINT 后得到 `task cancelled`，且没有半成品输出目录。
GPU 已提交的工作可能先执行完，取消不是任意 kernel 内部即时中断。

脱敏 JSON 见 [reports](reports)。原始 run.json 与可试听产物见 NO_TRACK/runs 下的
`leap-wgpu-30s-native`、`leap-wgpu-3s`、`leap-cpu-3s`、`deux-cpu-3s`。
Deux 的最终 WGPU 3 秒 smoke 总耗时 6.944 s；直接 MP3 截取 30–33 秒的 Leap WGPU
smoke 总耗时 5.748 s，输出均为精确 132300 samples / channel。
对应产物目录为 `deux-wgpu-3s` 与 `mp3-wgpu-3s`。
运行报告内保留实际模型、有效 config、张量数、权重和 PCM SHA256、stem origin、阶段计时与 RTF。
