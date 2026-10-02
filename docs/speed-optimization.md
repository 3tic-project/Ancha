# 推理速度审计与优化（2026-10-02）

本轮审计 Ancha 当前支持的全部模型：Leap Xe（BS-RoFormer）、Deux（Mel-Band）、
HyperACE v2 和四个经典 MDX ONNX，覆盖 CPU 与 WGPU。目标是在不改变分离上下文、精度
（FP32）和数值语义的前提下提速；所有优化都经过真实权重与固定参考实现的波形比对。

硬件与环境同前两轮：Intel i5-12400（6 核 12 线程）、64 GB、AMD Radeon RX 580 8 GB / Metal，
macOS 15.7.7 x86_64，Rust 1.92.0、Burn 0.21.0、CubeCL 0.10.0。基线是提交 `4221ed0` 的
release 二进制（`convert,wgpu,accelerate,onnx,cpu-opt`），对比二进制用相同特性构建。
对比矩阵与数值比对使用的二进制与本提交只差 MDX batch>1 的 patch 分批和一行 CLI 帮助文字
（两者都不经过 batch>1 路径）；profile 记录与 batch 复测使用本提交的构建。

## 审计发现

| 编号 | 现象 | 证据 | 根因 |
|---|---|---|---|
| A1 | WGPU 没有任何 autotune | 源码与特性检查 | 工作区以 `default-features=false` 依赖 Burn，未开启 `autotune`；matmul 用默认策略，卷积固定 Direct |
| A2 | MDX GPU 时间 65% 在 `DirectConv2dKernel`，14% 在布局拷贝 | CubeCL profile，KARA 2 30 秒 | RX 580 无 cooperative-matrix，CubeCL 卷积只剩直接卷积；即使 autotune 也无法选到 GEMM 卷积 |
| A3 | Leap 短块每个 chunk 约 2.17 万个 GPU kernel，稳态约 4.6 秒，平均每个约 0.2 ms | CubeCL profile | 默认 128/4 attention 分块使每层约 134 个 tile；90 个频带各自 norm / Linear / mask MLP；RoPE 用 narrow+neg+cat |
| A4 | CPU 默认 NdArray 卷积为直接卷积，Leap 短块 76 秒 | 实测 | NdArray 无 GEMM 卷积；Burn 0.21 已推荐 Flex 后端 |
| A5 | 换成 Flex 后 Leap 仍有 43% 时间在 RoPE、14% GELU、15% reshape 拷贝 | macOS `sample` 调用树 | Flex 逐元素算子单线程，非连续/非常见广播布局走逐元素 strided 迭代 |

## 已采用的优化

### 后端与内核选择

- **WGPU 启用 Burn autotune**。首次遇到新形状时实测候选内核并缓存（仓库内为 `target/autotune`）。
  首次运行明显变慢，报告里的 `timings.model_call_seconds[0]` 单独体现。
- **CPU 默认改用 Burn Flex**：gemm 矩阵乘、Rayon 并行、im2col 卷积、零拷贝 strided view。
  `--backend ndarray` 保留旧后端，用于消融；`cpu-opt` / `accelerate` 只影响它。
- **MDX GPU 卷积改写为 patch gather + GEMM**（`--conv-strategy auto`）：`unfold` 生成按
  (通道, ky, kx) 排列的 patch 矩阵，与 `[out,in,kh,kw]` 权重做一次 autotune GEMM。乘加项与原卷积相同，
  只改变累加顺序。CPU 上 Flex 原生 im2col 更快，因此 auto 只在 WGPU 启用。

### RoFormer 计算图（BS / Mel / HyperACE 共用）

- **加载期精确折叠**：RMSNorm 的 `sqrt(dim)·gamma` 折进其后投影的输入行（band split、q/k/v/gate、
  FFN 输入）；1/sqrt(head_dim) 折进 query 的 RoPE 表（head_dim=64 时是 2 的幂，逐位精确）。
- **RoPE 改写**：`x·cos + swap_pairs(x)·sin±`，相邻二元组交换用一次 select，成对符号预置在表中；
  每次 forward 每轴只建一次表。q/k/v 拆成三个连续输出的投影，避免拆头时的跨步拷贝。
- **同宽频带合批**：相同输入宽度的频带合成一组，band split 与 mask 末层按组做批量 GEMM；
  mask 隐层对全部频带做一次批量 GEMM。Leap 的 90 个频带只剩 7 组；Mel 的非相邻同宽频带
  用一次 gather 恢复原频带顺序，重叠频带仍按覆盖数平均。
- **复数 mask 在主机应用**：网络输出 `[stems, rows, re/im, frames]`，与原频谱的复数乘法在 iSTFT
  前完成，去掉设备上的多次 narrow / cat。
- **显式稠密化**：轴向转置和拆头后的张量只拷贝一次（CubeCL 对同形状 reshape 保持视图，
  否则每个消费者各拷一次）。
- **自适应 attention 分块**：未指定 `--query-tile` / `--group-tile` 时，在 `--max-score-mib`
  （默认 512）内选最大的精确分块：先保持整段 query，再尽量合并组。短块 Leap 的每 chunk GPU kernel
  从约 2.17 万降到约 1.1 千；原生块每层时间 attention 从 322 个 tile 降到 18 个。
- **CPU 分组并行**：轴向 Transformer 的序列组（时间轴为频带、频率轴为帧）彼此独立，
  `--host-threads`（CPU 默认逻辑核数，WGPU 固定 1）把组切给主机线程，各线程内算术完全相同；
  并发 worker 平分 score 预算，总量仍不超过 `--max-score-mib`。

## 数值一致性

最终二进制对固定参考实现全部通过（门槛 max_abs < 1e-3 且波形 SNR > 50 dB）：

| 模型 | 后端 | max_abs | 波形 SNR |
|---|---|---:|---:|
| Leap Xe voc | CPU (Flex) / WGPU | 2.16e-6 / 2.59e-6 | 116.42 / 114.99 dB |
| Deux vocals | CPU / WGPU | 1.65e-6 / 1.65e-6 | 124.57 / 123.88 dB |
| Deux instrumental | CPU / WGPU | 1.68e-6 / 1.59e-6 | 123.49 / 121.52 dB |
| HyperACE v2 voc | CPU / WGPU | 2.21e-6 / 8.82e-6 | 123.05 / 118.71 dB |
| HyperACE v2 inst | WGPU | 1.07e-5 | 115.83 dB |
| MDX 9482 / KARA | WGPU | 5.36e-7 / 3.58e-7 | 126.14 / 129.17 dB |
| MDX KARA 2 | CPU / WGPU | 6.56e-7 / 5.96e-7 | 126.73 / 127.13 dB |
| MDX Inst HQ 2 | CPU / WGPU | 8.94e-7 / 1.07e-6 | 123.30 / 122.28 dB |
| MDX KARA 2，30 秒 6 块 | WGPU | 8.34e-7 | 127.67 dB |

RoFormer 参考为固定 MSST / HyperACE PyTorch 2.2.2 forward，单个 132300-sample chunk；
MDX 参考为固定 UVR DSP + CPU ONNX Runtime 1.20.1。数值与上一轮报告同量级，说明折叠、合批、
分块与并行只改变了浮点调度。这是实现一致性，不是 SDR 或数据集质量结论。
明细见 [speed-parity.json](reports/speed-parity.json)。

## 速度

### 基线与优化对比（各自默认设置）

串行运行，同一 PCM、权重和音频上下文；表中为中位数（每侧 1–2 次）。WGPU autotune 已预热，
基线二进制没有 autotune。RoFormer 短块指 `--chunk-samples 132300 --overlap 1`，原生块为
manifest 上下文（Leap 881559、HyperACE 960000 samples）；MDX 都用 UVR 原生 chunk。
基线 CPU 为 NdArray + `cpu-opt` + Accelerate，优化后 CPU 为 Flex。

| 场景 | 后端 | 音频 / 块数 | 基线总耗时 | 优化后 | 加速 | 优化后 RTF |
|---|---|---|---:|---:|---:|---:|
| Leap 短块 | CPU | 3 秒 / 1 | 70.92 s | 12.19 s | 5.82× | 4.06 |
| Deux 短块 | CPU | 3 秒 / 1 | 42.49 s | 8.90 s | 4.77× | 2.97 |
| HyperACE voc 短块 | CPU | 3 秒 / 1 | 40.95 s | 9.32 s | 4.39× | 3.11 |
| MDX KARA 2 | CPU | 3 秒 / 1 | 12.83 s | 4.14 s | 3.10× | 1.38 |
| MDX KARA 2 | CPU | 30 秒 / 6 | 79.42 s | 25.84 s | 3.07× | 0.86 |
| MDX Inst HQ 2 | CPU | 3 秒 / 1 | 21.51 s | 6.68 s | 3.22× | 2.23 |
| Leap 短块 | WGPU | 30 秒 / 10 | 45.14 s | 38.26 s | 1.18× | 1.28 |
| Leap 原生块 | WGPU | 3 秒 / 1 | 70.48 s | 66.82 s | 1.05× | 22.27 |
| Deux 短块 | WGPU | 30 秒 / 10 | 29.93 s | 30.67 s | 0.98× | 1.02 |
| HyperACE voc 短块 | WGPU | 30 秒 / 10 | 46.54 s | 28.12 s | 1.66× | 0.94 |
| HyperACE voc 原生块 | WGPU | 3 秒 / 1 | 50.14 s | 46.07 s | 1.09× | 15.36 |
| MDX KARA 2 | WGPU | 30 秒 / 6 | 14.52 s | 7.25 s | 2.00× | 0.24 |
| MDX Inst HQ 2 | WGPU | 30 秒 / 6 | 21.62 s | 10.96 s | 1.97× | 0.37 |
| MDX KARA | WGPU | 30 秒 / 6 | 6.66 s | 4.53 s | 1.47× | 0.15 |
| MDX 9482 | WGPU | 30 秒 / 6 | 6.68 s | 4.57 s | 1.46× | 0.15 |

总耗时包含加载、解码、DSP、推理和 WAV 写出。CPU 增益来自两部分：同一新计算图下旧 NdArray
后端的 Leap 短块为 21.25 秒（比基线快 3.3×），换成 Flex 再到 12.19 秒。
WGPU 的 RoFormer 已基本受矩阵乘吞吐限制：Deux 模型阶段 27.49 → 27.56 秒持平，总耗时因加载期
权重堆叠多约 0.7 秒；原生长块的大部分时间也在 GEMM。明细见 [speed-benchmark.json](reports/speed-benchmark.json)。

## 当前 profile 记录

以下为本提交默认设置在本机的稳态记录，可作为迁移到其它设备（例如 CUDA）时的对照。
WGPU 的首次调用额外包含内核编译与 autotune 缓存查询；缓存为空时更长（HyperACE 原生块首次约 255 秒）。
CPU 默认 `--host-threads 12`（本机逻辑核数）。

| 模型 / 上下文 | 设备 | 每次调用（首次 / 稳态） | 有效 attention 分块（时间轴 组×query / 频率轴） | 其它设置 |
|---|---|---|---|---|
| Leap 短块 | CPU | 11.66 s / — | 8×259 / 22×90（每 worker） | host_threads=12 |
| Leap 短块 | WGPU | 3.91 s / 3.70 s | 90×259 / 259×90 | — |
| Leap 原生块 | WGPU | 65.95 s / — | 5×1722 / 1722×90 | — |
| Deux 短块 | CPU | 7.32 s / — | 5×301 / 26×60（每 worker） | 双输出头 |
| Deux 短块 | WGPU | 3.60 s / 2.66 s | 60×301 / 301×60 | 双输出头 |
| HyperACE voc 短块 | CPU | 8.78 s / — | 6×259 / 22×62（每 worker） | — |
| HyperACE voc 短块 | WGPU | 3.00 s / 2.67 s | 62×259 / 259×62 | — |
| HyperACE voc 原生块 | WGPU | 45.08 s / — | 4×1876 / 1876×62 | — |
| MDX KARA 2 | CPU | 4.09 s / 4.29 s | — | Flex 卷积 |
| MDX KARA 2 | WGPU | 1.21 s / 1.10 s | — | GEMM 卷积 |
| MDX Inst HQ 2 | WGPU | 1.83 s / 1.68 s | — | GEMM 卷积 |
| MDX KARA / 9482 | WGPU | 0.79–0.81 s / 0.64 s | — | GEMM 卷积 |

每次调用为一个 RoFormer chunk 或一个 MDX batch，包含上传与读回。热点占比（3 秒单块）：

| 模型 | WGPU 内核（数量；占比） | CPU 采样（忽略空闲等待） |
|---|---|---|
| Leap 短块 | 1146；MatMul 70.6%、未融合二元 8.5%、融合逐元素 6.9%、拷贝 5.1% | GEMM 33.3%、拷贝/布局 30.6%、逐元素 10.9%、erf 7.4% |
| Deux 短块 | 2098；MatMul 58.9%、未融合二元 11.1%、归约 7.3% | 拷贝/布局 32.2%、GEMM 30.2%、逐元素 11.7% |
| HyperACE voc 短块 | 2449；MatMul 48.5%、归约 15.7%、逐元素 8.6%、直接卷积 7.0% | GEMM 33.9%、拷贝/布局 29.8%、逐元素 10.2% |
| MDX KARA 2 | 243；MatMul 66.7%、patch 拷贝 15.6%、逐元素 7.3% | GEMM 61.1%、其它 16.8%、拷贝 11.3% |
| MDX Inst HQ 2 | 243；MatMul 65.4%、patch 拷贝 16.6%、逐元素 7.4% | — |

WGPU 占比来自 CubeCL 逐内核同步 profile（绝对时间被放大，只看比例）；CPU 占比来自 macOS `sample`
的栈顶统计，含开头的模型加载。原始数据见 [speed-profile.json](reports/speed-profile.json)。
下一步的明显方向：WGPU 上提高 GEMM 吞吐（RX 580 的 unit matmul 约为峰值的 10%），CPU 上减少约三成的
布局拷贝；均需另行验证数值。

MDX `--mdx-batch-size` 在 GEMM 卷积下逐 batch 项生成 patch 矩阵，避免 3×3 层 9 倍展开超出 8 GB 显存。
Inst HQ 2 30 秒在 batch 1 / 2 / 4 下为 10.76 / 10.45 / 10.71 秒；修正前 batch 4 因显存压力为 116.3 秒。

## 试验后未采用

| 方案 | 结果 | 处理 |
|---|---|---|
| Burn 融合 attention 算子（WGPU flash / CPU Flex） | RX 580 上无稳定收益；Flex 版本单线程逐头执行 | 未保留 |
| Metal MSL 直编（`burn/metal`） | 与 WGSL 路径相比无可测收益 | 未保留 |
| `CUBECL_AUTOTUNE_LEVEL=full` | 稳态无收益，首次调优更久 | 保持默认 balanced |
| CPU 上 patch gather + GEMM 卷积 | 明显慢于 Flex 原生 im2col | auto 只在 WGPU 使用 |
| HyperACE SegmModel 卷积改写 | GPU 直接卷积只占 4.7% 时间 | 未改 |
| CPU q/k/v 保持 strided 视图 | Leap 短块各两次 13.7–13.9 秒对 14.3–15.6 秒，但 WGPU 需要稠密 | 统一稠密化，保持单一路径 |

## 边界

- 速度数字来自单机、单首歌的固定片段，开发机同时有其它桌面负载；中位数降低了但没有消除噪声。
- WGPU 首次运行某个模型/上下文要做 autotune，HyperACE 原生块首次约 255 秒，之后复用缓存；
  缓存随 Burn/CubeCL 版本或驱动变化可能失效。
- 没有 profiler 级显存峰值；`--max-score-mib` 只约束 score 张量。
- 本轮未验证 CUDA、Linux / Windows GPU、其它 CPU；CPU 线程数默认取逻辑核数，超线程收益与机器有关。
  CUDA 后端与 Linux（Xeon + Tesla P4）上的三后端实测见后续的 [CUDA 记录](cuda.md)。
- 未改变模型上下文、精度或量化；更快的低精度、蒸馏和更短上下文都属于不同的质量配置。

## 复现

```bash
cargo build --release --locked --features convert,wgpu,accelerate,onnx,cpu-opt
ANCHA_OLD=<4221ed0 二进制> ANCHA_NEW=target/release/ancha bash scripts/benchmark-matrix.sh
```

脚本先预热 autotune，再串行、交替顺序运行基线与优化二进制，最后用
`scripts/summarize_benchmarks.py` 汇总中位数。本轮数字由同样命令顺序的 NO_TRACK 开发脚本产生。
