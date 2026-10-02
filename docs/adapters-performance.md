# HyperACE / MDX 本机适配与性能

2026-10-02，Intel i5-12400（6核/12线程）、64GB RAM、RX580 8GB，macOS15.7.7 x86_64。
Rust1.92.0、Burn0.21.0、CubeCL0.10.0、onnx-rs0.1.2。使用本地歌曲第30秒起的同一3秒/30秒
float32 WAV，44100Hz stereo；原始音频、权重、完整日志和产物均在 NO_TRACK。

## 跑通与实现一致性

HyperACE v2 voc/inst 各严格加载1170个张量，保留完整 SegmModel 与 MLP mask。
GPU 使用 RX580 / Metal / WGPU FP32；CPU 为 NdArray，macOS BLAS=Accelerate。
以下不是干净源 SDR，而是与固定原版实现的最终波形差异。门槛 max_abs<1e-3 且SNR>50dB。

| 模型 / 测试 | max_abs | waveform SNR |
|---|---:|---:|
| HyperACE voc / CPU / 3s短上下文 | 2.4140e-06 | 123.93 dB |
| HyperACE voc / CPU-opt / 3s短上下文 | 4.5896e-06 | 121.97 dB |
| HyperACE voc / WGPU / 3s短上下文 | 8.5533e-06 | 118.01 dB |
| HyperACE inst / WGPU / 3s短上下文 | 1.0267e-05 | 115.45 dB |
| MDX9482 / WGPU / 3s | 5.3644e-07 | 125.80 dB |
| MDX KARA / WGPU / 3s | 3.7998e-07 | 128.04 dB |
| MDX KARA2 / WGPU / 3s | 7.1526e-07 | 125.48 dB |
| MDX HQ2 / WGPU / 3s | 8.9407e-07 | 120.54 dB |
| MDX KARA2 / CPU / 3s | 5.9605e-07 | 127.14 dB |
| MDX KARA2 / CPU-opt 4 / 3s | 7.1526e-07 | 126.50 dB |
| MDX KARA2 / WGPU denoise+overlap0.5+batch2 | 6.5565e-07 | 128.04 dB |
| MDX KARA2 / WGPU batch2 / 30s | 1.0133e-06 | 126.61 dB |
| MDX HQ2 / WGPU batch2 / 30s | 2.2054e-06 | 122.04 dB |

HyperACE 短上下文为 chunk132300 / overlap1，明确不同于原生960000 / overlap4。
另外完成原生上下文的3秒输入执行：1876 frames、1chunk，总耗时58.185秒。
该原生 smoke 期间有开发编译，不是稳定吞吐测试；不能据短上下文时间声称原生模型实时。
CPU-opt HyperACE 短上下文单次42.207秒；早期CPU/WGPU smoke亦有并行开发工作，不用于加速倍数。

MDX 3秒测试保留261120-sample / T256原生网络上下文；30秒测试保留默认step/Hann/compensate。
30秒两模型输出均为精确1323000 samples/channel，残差重建 max_abs=5.96e-8。
原版UVR参考直接调用固定 separate.py 的方法和 STFT，由CPU ORT1.20.1执行；session创建前关闭telemetry。
30秒、denoise和多块重叠最终输出均通过比对。

## WGPU 图优化：固定同一二进制 / PCM / 参数

编译和测试结束后串行执行，基线/优化顺序交替。每个3秒设置3次，取中位数；
总耗时包含每次新进程的载入、DSP和写出。基线关闭BN预计算/折叠与无贡献尾块跳过。
推理上下文、denoise=false、默认step、batch1、FP32均一致。

| 模型 | 基线总耗时 | 优化总耗时 | 总任务加速 | 模型阶段加速 |
|---|---:|---:|---:|---:|
| KARA2 / 3秒 | 5.491s | 2.918s | 1.88× | 1.97× |
| Inst HQ2 / 3秒 | 7.448s | 4.064s | 1.83× | 1.90× |

3秒音频：原始流程2次forward，优化1次；5个ConvTranspose后BN被安全折叠，其余22个TDF BN预计算。
节省的尾块不与最终音频交叠；所有有贡献的块仍完整执行。输出差异均低于1e-6。
这些倍数不能直接推广到长曲，因为长曲中无贡献尾块占比更小。

30秒同上下文的一组对照：17.374s → 14.479s，
总任务1.20×，7次forward→6次。该对照只有一次，不称为稳定中位数。
KARA2本机30秒实测RTF约0.483；HQ2 batch2单次22.469s，RTF0.749。
这两项本地观测低于RTF1，不代表所有曲目、机器或任务都达到实时。

## CPU SIMD / 并行卷积

同一KARA2原生3秒任务、同一优化图与PCM，固定VECLIB_MAXIMUM_THREADS=1。
基线未编入SIMD/并行卷积；cpu-opt启用Burn SIMD与NdArray/Rayon多线程。
三种设置在开发编译结束后串行执行，每种一次，thread值是启动设置而非监测到的进程线程总数。

| 构建 / Rayon设置 | 总耗时 | 模型阶段 | 相对基线 |
|---|---:|---:|---:|
| 普通构建 / 1 | 76.432s | 76.261s | 1.00× |
| cpu-opt / 1 | 58.255s | 58.074s | 1.31× |
| cpu-opt / 4 | 17.704s | 17.526s | 4.32× |

优化波形与基线 max_abs=5.36e-7；cpu-opt4另通过原版UVR比对。保留每种构建的feature、binary SHA256、
启动线程环境和真实耗时。这个CPU倍数来自一次观察，不能当作其它后端或HyperACE的加速倍数。

## 批量：正确但本机未提速

KARA2同一30秒：batch1为14.889s，batch2为15.300s；两轨逐采样相同。
6块从6次forward变为3次，但本机约慢2.8%，因此默认保持batch1。batch2/4作为显式选项，
需在目标设备测量，增加显存；不会因为forward次数更少就宣传更快。
HQ2 batch2的30秒运行和最终波形另行通过UVR比对，未证明它比HQ2 batch1更快。

## 验收与边界

28项Rust测试、fmt、Clippy -D warnings、包含convert/accelerate/wgpu/onnx/cpu-opt的Release构建通过。
另一个不含Apple BLAS/WGPU的CPU配置28项测试通过；CI增加ONNX与CPU优化编译/测试，远端CI尚未运行。
新增测试覆盖空间归一化、奇数resize、shuffle、HyperACE配置、图调度/BN折叠/拒绝规则、batch轴、
MDX复数打包/低频清零/高频补零，以及错误/取消不发布输出。

未完成CUDA、HyperACE inst独立CPU波形验收、整首277秒、干净源SDR与大数据集回归、其它硬件平台。
没有profiler实测峰值VRAM；主机仍保留受样本预算限制的PCM/OLA，没有磁盘流式实现。
短时间测量包含驱动/缓存状态影响，未分别统计冷启动编译和常驻服务稳态。
早期并行编译期间的临时计时只保留在NO_TRACK，不纳入上述串行加速表。

使用与复现见 [适配文档](adapters.md)，数值与binary摘要见 [reports](reports)。
试听文件在 NO_TRACK/runs 的 `mdx-kara-2-wgpu-30s-b1` / `mdx-hq-2-wgpu-30s-b2` /
`hyperace-wgpu-3s` / `hyperace-inst-wgpu-3s`；文件、模型和参考环境均未进入Git。
