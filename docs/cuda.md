# CUDA 后端适配与优化（2026-10-02）

本轮把迁移包（提交 `f55bfb1`）搬到一台 Linux + NVIDIA 机器上，新增 CUDA 后端，并在不改变分离上下文、
FP32 精度和数值语义的前提下做 CUDA 侧优化。全部 8 个模型（9 个输出）在 CUDA 上通过与之前相同的
PyTorch / UVR 参考比对；所有速度数字都来自本机串行实测，profile 来自 Nsight Systems。
第二轮用手写 CubeCL 内核替换 attention、投影 GEMM 与卷积（[手写 CUDA 内核](#手写-cuda-内核第二轮)），
RoFormer 原生块再快 3.1–3.8×，整首歌全部模型从 61 分钟降到约 20 分钟。

## 环境

| 项 | 本机 |
|---|---|
| CPU / 内存 | Intel Xeon E5-2673 v3（12 核 24 线程，无 SHA 指令扩展），15 GB |
| GPU | NVIDIA Tesla P4 8 GB，Pascal sm_61，无 Tensor Core；75 W，实测运行在 1113 MHz 应用时钟 |
| 驱动 / 工具链 | 535.288.01、CUDA 12.2（NVRTC 动态加载）；Rocky Linux 9.2；Rust 1.92.0、Burn 0.21.0、CubeCL 0.10.0 |
| WGPU | Vulkan（同一块 Tesla P4，NVIDIA ICD） |
| 参考环境 | Python 3.11、PyTorch 2.2.2+cpu、ONNX Runtime 1.20.1（`NO_TRACK/.venv-parity`） |

## 构建与使用

```bash
# nvcc 只用于构建时确定 cudarc 的 CUDA 绑定版本；没有 nvcc 时设 CUDARC_CUDA_VERSION=12020 一类的值。
# 默认 feature 已含 CPU / WGPU / ONNX，只需追加 cuda。
PATH=/usr/local/cuda/bin:$PATH cargo build --release --locked --features cuda --examples
target/release/ancha doctor --backend cuda          # 打印 GPU 名称、算力和显存后做 2×2 matmul
target/release/ancha separate NO_TRACK/runs/clip-3s.wav --model NO_TRACK/models/leap-xe-voc \
  --backend cuda --device 0 --output NO_TRACK/runs/my-cuda
```

运行时只需要驱动（libcuda）和 NVRTC（libnvrtc），不链接 CUDA 库，也不依赖 cuBLAS / cuDNN。
设备序号不存在、驱动缺失会在 CubeCL 启动前直接报错（退出码 2），不再出现设备线程 panic。

## 实现

| 部分 | 内容 |
|---|---|
| feature 与 CLI | 根包 `cuda` feature 打开 Burn `cuda`、fusion、autotune；`--backend cuda --device N`，报告 `backend=cuda` |
| GPU 默认值 | `--host-threads` 固定 1；MDX `--conv-strategy auto` 在 WGPU 与 CUDA 都选 patch gather + GEMM；CUDA 上 HyperACE SegmModel 的非分组卷积同样走 GEMM；RoFormer `--linear-layout auto` 在 CUDA 选合并投影 |
| 设备预检 `src/cuda.rs` | 用 cudarc 驱动 API 校验序号，读取名称、算力、显存 |
| 内核缓存 | 默认把 NVRTC 生成的 PTX 存到缓存根下 `ptx-sm<算力>/<模型文件名>/`（仓库内为 `target/`，仓库外为用户缓存目录 `cubecl/`）；cubecl.toml / Burn.toml 设置了 `[compilation] cache` 时尊重用户配置 |
| 设备故障检查 `src/device.rs` | 记录 CubeCL 设备线程（`DSU-*` / `DSD-*`）的 panic，在加载后与每次模型调用后检查，失败即报错且不发布结果 |
| 加载 | RoFormer 权重只读一次，SHA-256 在另一线程与解析、折叠、上传并行，摘要不符仍报 checksum 错误 |
| 手写内核 `crates/ancha-models/src/fused` | 单遍 attention、带收尾的 GEMM、隐式 GEMM 卷积、NCHW InstanceNorm，以 Burn fusion 自定义算子接入，只在 CUDA 分派；见下文第二轮。裸 GEMM 在 `m` 不是 128 的倍数时零填充到 128 列 |
| 测试 | `tests/cuda_contracts.rs`（微型 BS / Mel 整条分离 CUDA 对 Flex < 1e-5，其中一组 64 宽 head 的模型走手写内核；attention / 门控 attention 对分块实现、GEMM 与各收尾对 Burn 算子、两种 GEMM 卷积对 conv2d，含不整块、跨步输入与不支持的形状；无效序号）与 `tests/cuda_device_failure.rs`（超显存分配必须被报告），需 `--features cuda` 与真实 GPU；CI 只 `cargo check` |

### 显存不足曾静默产出错误结果

CubeCL 0.10 在设备服务线程里分配显存并直接 unwrap。显存不足时 panic 只发生在该线程，主线程之后的读回拿到
旧缓冲，`Backend::sync` 也返回成功。原生 Leap 块把 `--max-score-mib` 调到 1024 / 2048 时：

| `--max-score-mib` | 时间轴分块 | 模型调用 | vocals RMS | 设备线程 panic |
|---:|---|---:|---:|---:|
| 256 | 2 组 × 1722 | 28.95 s | 0.1826 | 0 |
| 512（默认） | 5 组 × 1722 | 28.50 s | 0.1826 | 0 |
| 1024 | 11 组 × 1722 | 28.31 s | 0.18239 | 80 |
| 2048 | 22 组 × 1722 | 12.99 s | 0.09234 | 512 |

同一块 CPU（Flex）结果为 vocals RMS 0.1826、峰值 1.0577。修复前后两行以退出码 0 发布了错误的 stems，
2048 的“提速”只是 attention 没有被计算。现在两者都以“GPU device thread failed … out of device memory”失败，
输出目录不出现。走分块 attention 时（`--attention-kernel tiled`，或非 CUDA 后端）8 GB 显卡上原生 RoFormer 块应保持默认 512；
CUDA 默认的单遍 attention 不分配 score 张量。

## 数值一致性

与 [速度优化记录](speed-optimization.md) 同一门槛（max_abs < 1e-3 且波形 SNR > 50 dB）、同一 3 秒片段：
RoFormer 为单个 132300-sample chunk 对固定 MSST / HyperACE PyTorch forward，MDX 为原生 chunk 对固定
UVR DSP + CPU ONNX Runtime。下表为 CUDA 最终默认设置（`c252cdf`：合并投影、MDX 与 HyperACE 的 GEMM 卷积）：

| 模型 | 输出 | max_abs | 波形 SNR |
|---|---|---:|---:|
| Leap Xe voc | vocals | 6.36e-6 | 107.60 dB |
| Deux | vocals / instrumental | 8.34e-7 / 8.94e-7 | 125.40 / 122.01 dB |
| HyperACE v2 voc | vocals | 5.90e-6 | 120.47 dB |
| HyperACE v2 inst | instrumental | 7.53e-6 | 117.30 dB |
| MDX 9482 | all_vocals | 5.96e-7 | 126.40 dB |
| MDX KARA | lead_vocals | 3.87e-7 | 129.36 dB |
| MDX KARA 2 | karaoke_mix | 6.56e-7 | 127.05 dB |
| MDX Inst HQ 2 | instrumental | 9.46e-7 | 122.50 dB |

批量投影布局下 Leap / HyperACE voc 的数值与上表相同，Deux 为 8.05e-7 / 8.94e-7（125.58 / 122.19 dB），
HyperACE inst 用 Burn conv2d 时为 7.51e-6（117.29 dB）。
CubeCL CUDA 以 fast-math 选项调用 NVRTC；Leap 的 107.6 dB 低于此前 RX 580 WGPU 的 115.0 dB，仍远高于门槛。
这是实现一致性，不是 SDR 或质量结论。明细见 [cuda-parity.json](reports/cuda-parity.json)。

同一台机器上 CPU（Flex）全部 8 个模型通过（Leap 1.82e-6 / 120.83 dB 等）。WGPU（Vulkan，同一块 P4）
**Deux 与 HyperACE inst 未通过**：Deux vocals max_abs 3.23e-2、SNR 40.83 dB，HyperACE inst 8.27e-3、40.38 dB，
HyperACE voc 虽过门槛但只有 78.56 dB；Leap 与四个 MDX 正常。偏差是确定性的，与 CUDA 之前的二进制
`f55bfb1` 逐位一致，与投影布局、attention 分块无关，也没有设备线程 panic；可重叠频带的 scatter-add
在该设备上与 CPU 完全一致。因此这是已有的 WGPU / NVIDIA Vulkan 问题，不是本轮引入；NVIDIA GPU 上应使用 CUDA。
明细见 [linux-parity.json](reports/linux-parity.json)。

## CUDA 侧优化与消融

| 改动 | 场景 | 之前 | 之后 |
|---|---|---:|---:|
| 持久化 PTX 缓存 | KARA 2，1 块，autotune 已缓存 | 模型调用 12.9–14.7 s | 0.65 s |
| PTX 缓存按模型分区 | KARA 2 加载 / Leap 短块加载 | 1.75 s / 3.24 s | 0.57 s / 2.11 s |
| MDX GEMM 卷积（保持默认） | KARA 2 模型调用 | Burn conv2d 1.71 s | 0.65 s |
| CUDA 合并投影 | Deux / HyperACE / Leap 短块模型调用 | 1.95 / 2.11 / 2.71 s | 1.78 / 1.96 / 2.63 s |
| CUDA 合并投影 | HyperACE / Leap 原生块模型调用 | 20.7 / 28.5 s | 19.9 / 28.6 s |
| 哈希与上传重叠 | Deux / Leap 加载 | 6.4–6.6 / 2.11 s | 5.4 / 1.66 s |
| HyperACE SegmModel GEMM 卷积 | HyperACE 短块 / 原生块模型调用 | 1.96–1.99 / 19.9 s | 1.75 / 17.9 s |

- PTX 缓存：CubeCL CUDA 默认没有编译缓存，每个进程都要用 NVRTC 重新编译所有内核；autotune 结果虽已缓存，
  KARA 2 的 3 秒任务仍要 13–15 秒。新分区第一次运行仍会编译一次（KARA 2 约 12 秒、Leap 短块约 27 秒）。
- 分区：CubeCL 启动时把整个 CBOR 缓存文件反序列化（全部模型 68 MB 时约 1.2 秒），按模型文件名分区后只读本模型的条目。
- 卷积：Pascal 没有 Tensor Core，CubeCL 只能走直接卷积，patch gather + GEMM 快 2.6×，与 RX 580 结论一致。
- 合并投影：CUDA 上不慢于批量投影，因此 `auto` 在 CUDA 选它；CPU / WGPU 维持原默认。
- 加载：本机 sha2 只有约 160 MB/s，830 MB 的 Deux 仅哈希就约 5.2 秒，重叠后仍以哈希为下限。
- HyperACE：profile 显示 17 次直接卷积占 GPU 时间 13.9%（短块）/ 9.4%（原生块）。非分组卷积复用 MDX 的
  GEMM 改写后快约 10%，深度可分离卷积仍用 Burn conv2d。`--conv-strategy` 现在也作用于 HyperACE，
  `auto` 只在 CUDA 上选 GEMM，CPU / WGPU 行为不变；run.json 的 `conv_strategy` 记录实际路径。

明细见 [cuda-ablation.json](reports/cuda-ablation.json)。

## 手写 CUDA 内核（第二轮）

第一轮 profile 显示原生块里 MatMul 占 48–64%，softmax 前后对 score 张量的显存往返又占三分之一以上，而
Pascal 上 CubeCL 的通用矩阵乘只有 cuBLAS 的约 1/3。第二轮用 CubeCL 手写 FP32 内核（`crates/ancha-models/src/fused`），
以 Burn fusion 自定义算子接入，模型代码仍对 `B: Backend` 泛型，其它后端和不支持的形状自动回到原 Burn 路径。

| 内核 | 做法 | 算子级实测（Tesla P4） |
|---|---|---|
| 单遍 attention | 每个 cube 处理一个 (group, head) 的 64 个 query，按 64 个 key 一块流式读，在线 softmax，score 不出寄存器 / 共享内存；128 线程，每线程 4 行 × 8 列 | Leap 时间轴 90×1722 token：分块物化 1.79 s → 0.200 s（2.73 TFLOPS），最大差 1.3e-6 |
| GEMM | 128×128 输出块、k 步长 8、寄存器中转的双缓冲共享内存，Aᵀ 存放使每种收尾都不超过 128 寄存器（每 SM 两个 cube） | 155k 行投影 3.7–4.1 TFLOPS，Burn 1.3–1.8 TFLOPS，与第一轮测得的 cuBLAS（3.6–4.4）相当 |
| 隐式 GEMM 卷积 | 装载每个 k 块时从输入直接取 patch，不再物化 patch 矩阵（HyperACE 全分辨率层原需约 1 GB）；按 16/48/64 通道取块，卷积几何为编译期参数 | MDX 输出与原 GEMM 卷积逐位一致 |

在此之上把整个 RoFormer 块重排成 5 次内核调用：q / k / v / gate 一次 GEMM（RoPE 在其收尾中完成）、
带 sigmoid 门控的 attention 直接读这块打包投影、输出投影与前馈两层 GEMM 在收尾中加 bias、做 erf-GELU 和残差。
旋转、门控、残差不再各自扫一遍显存，score 张量也不再分配，所以 `--max-score-mib` 对 CUDA 默认路径不起作用
（run.json 的 score 估计为 0）。

原生块单次模型调用（3 秒片段，缓存已预热）随各步变化：

| 提交 | 改动 | Leap | Deux | HyperACE voc | MDX HQ 2 | MDX KARA 2 | MDX 9482 |
|---|---|---:|---:|---:|---:|---:|---:|
| `404ec9c` | 第一轮结束 | 28.50 s | 11.68 s | 17.94 s | 0.96 s | 0.62 s | 0.40 s |
| `7eaee28` | 单遍 attention + 投影 GEMM | 10.08 s | 4.26 s | 7.81 s | | | |
| `f644f5d` | 整块重排与收尾融合 | 7.88 s | 3.34 s | 6.57 s | | | |
| `5d5af29` | attention 改 4×8 寄存器块 | 7.59 s | 3.27 s | 6.44 s | | | |
| `8d1f14d` / `d8ac0d2` | 隐式 GEMM 卷积及调优 | 7.59 s | 3.27 s | 5.70 s | 0.55 s | 0.37 s | 0.26 s |

Leap、Deux、HyperACE 分别快 3.8×、3.6×、3.1×，MDX 快 1.5–1.7×。只有 attention 一项时 Leap 为 13.41 s。

- 开关：`--attention-kernel auto|fused|tiled`、`--gemm-kernel auto|custom|burn`，`auto` 只在 CUDA 启用；
  run.json 记录 `attention_kernel` / `gemm_kernel`。`tiled` + `burn` 即第一轮路径（Leap 原生块 28.54 s），用于消融。
  卷积内核在 CUDA 上随 `--conv-strategy gemm`（默认）启用，深度 `c·kh·kw` 不是 8 的倍数时（如 MDX 首层）走原 GEMM 卷积。
- 只在 CUDA 上分派：同一批内核在 WGPU（Vulkan，同一块 P4）上输出错误（Deux 片段 SDR −5 dB），CubeCL 的 WGPU 目标不支持
  GEMM 所用的共享内存向量重解释，因此显式请求时 CLI 报错，WGPU 行为与之前逐位相同。
- 寄存器决定占用率：GEMM 超过 128 个寄存器时每个 SM 只能放一个 cube，同样的算术慢约 1.6 倍（profile 中
  `registersPerThread` 144 对 128）；改为 Aᵀ 布局后所有收尾变体都是 122–126 个。attention 受 48 KiB 共享内存限制，
  每 SM 两个 cube，因此 128 线程版本可用到 255 个寄存器（实际 204、无溢出）。
- 数值：单个内核对 Burn 实现的差异在 1e-6 量级（契约测试门槛 1e-5）。`d8ac0d2` 重跑参考比对仍为 8/8：
  Leap 6.51e-6 / 107.28 dB，Deux 7.60e-7 / 1.07e-6（126.04 / 122.23 dB），HyperACE voc 3.34e-6 / 121.52 dB，
  HyperACE inst 4.35e-6 / 118.56 dB，四个 MDX 与上表逐位相同。

迁移包中的第二轮参考比对（8/8）与整曲运行汇总已补入
[cuda-kernels.json](reports/cuda-kernels.json)，保留原始报告摘要与二进制 SHA；算子级微基准仅有本节记录，
迁移包未提供对应的结构化报告。汇总未记录构建提交，不能把整曲结果直接视为 `d8ac0d2` 的单次测速。

## 速度（Linux 三后端）

本节为第一轮（手写内核之前）的三后端对比；CUDA 当前数字见上一节与下面的整首歌示例。

同一二进制（`fa6efbf`）、同一 PCM / 权重 / 上下文，串行运行；GPU 每行先预热一遍，再测两遍取中位数。
WGPU 与 CUDA 跑在同一块 Tesla P4 上（WGPU 走 Vulkan），CPU 为 Burn Flex（24 个主机线程）。
RoFormer 短块为 `--chunk-samples 132300 --overlap 1`，原生块为 manifest 上下文；MDX 用原生 chunk。

| 场景 | 音频 / 块数 | CUDA 总耗时 | WGPU 总耗时 | CUDA 每块（首次 / 稳态） | WGPU 每块（首次 / 稳态） | CUDA RTF |
|---|---|---:|---:|---:|---:|---:|
| Leap 短块 | 30 秒 / 10 | 27.64 s | 39.05 s | 2.65 / 2.55 s | 7.96 / 3.22 s | 0.92 |
| Deux 短块 | 30 秒 / 10 | 22.62 s | 31.41 s | 1.80 / 1.65 s | 4.52 / 2.33 s | 0.75 |
| HyperACE voc 短块 | 30 秒 / 10 | 18.43 s\* | 34.61 s | 1.75 / 1.61 s\* | 11.57 / 2.32 s | 0.61\* |
| Leap 原生块 | 3 秒 / 1 | 30.43 s | 36.30 s | 28.58 s | 34.43 s | 10.14 |
| HyperACE voc 原生块 | 3 秒 / 1 | 19.91 s\* | 30.12 s | 17.94 s\* | 28.14 s | 6.64\* |
| MDX KARA 2 | 30 秒 / 6 | 4.88 s | 8.43 s | 0.65 / 0.62 s | 2.58 / 0.96 s | 0.16 |
| MDX Inst HQ 2 | 30 秒 / 6 | 7.01 s | 11.10 s | 0.98 / 0.95 s | 2.53 / 1.47 s | 0.23 |
| MDX KARA | 30 秒 / 6 | 3.38 s | 5.44 s | 0.43 / 0.40 s | 1.56 / 0.60 s | 0.11 |
| MDX 9482 | 30 秒 / 6 | 3.37 s | 5.29 s | 0.43 / 0.40 s | 1.41 / 0.60 s | 0.11 |

\* HyperACE 的 CUDA 行为加入 SegmModel GEMM 卷积后（`c252cdf`）的补测，各 2 次；矩阵中 `fa6efbf` 为
20.54 / 21.81 秒（每块 1.97 / 1.83 与 19.86 秒）。其余行的二进制在这些工作负载上与 `c252cdf` 行为相同。

| CPU（Flex） | 音频 / 块数 | 加载 | 模型 | 总耗时 | RTF |
|---|---|---:|---:|---:|---:|
| Leap 短块 | 3 秒 / 1 | 1.53 s | 13.18 s | 14.75 s | 4.92 |
| Deux 短块 | 3 秒 / 1 | 4.97 s | 9.33 s | 14.35 s | 4.78 |
| HyperACE voc 短块 | 3 秒 / 1 | 1.65 s | 9.90 s | 11.59 s | 3.86 |
| MDX KARA 2 | 30 秒 / 6 | 0.37 s | 39.15 s | 40.01 s | 1.33 |
| MDX Inst HQ 2 | 3 秒 / 1 | 0.46 s | 9.34 s | 9.88 s | 3.29 |

- 同一块 GPU 上 CUDA 的稳态每块比 WGPU 快 1.2–1.6×，总耗时快 1.2–1.9×。WGPU 即使 autotune 已缓存，每个进程
  首块仍要重建着色器管线（HyperACE 短块 11.6 秒对稳态 2.3 秒）；CUDA 读取 PTX 缓存，首块与稳态接近。
  注意本机 WGPU 的 Deux 与 HyperACE inst 输出未通过参考比对（见上节），这两行的 WGPU 数字只作耗时参考。
- MDX KARA 2 的 30 秒任务 CUDA 比本机 CPU 快 8.2×。CPU 原生 Leap 块（3 秒）单次约 197 秒（当时与一次编译并行，
  仅作量级参考），CUDA 为 28.6 秒。
- 30 秒短块的总耗时里，Deux 有 5.4 秒是加载（主要是 SHA-256），见上节。
- 与 CUDA 之前的二进制（`f55bfb1`）比，本机 CPU 的 Leap 短块 15.05–15.16 → 14.75 秒（只有加载变快），
  KARA 2 30 秒 40.04–40.15 → 40.01 秒，CPU 路径没有回退。
- 一次 WGPU 9482 运行在发布结果之后，于 NVIDIA Vulkan 驱动线程（`[vkrt] Analysis`，libEGL_nvidia 535）中段错误退出；
  结果完整，CUDA 运行未见此问题。

明细见 [cuda-benchmark.json](reports/cuda-benchmark.json)。

## 整首歌全模型示例

[examples/separate_all.rs](../examples/separate_all.rs) 在一个进程里用同一后端依次运行 8 个模型
（各自的原生上下文与该后端的默认设置），每个模型写出 stem 与 run.json，并汇总到 summary.json。

- 输入：`NO_TRACK/test_file/ReoNa - Amore.mp3`（11.2 MB，277.57 秒）。
- 二进制：`404ec9c`，`cargo build --release --locked --features cuda --examples`，即默认 feature 加 cuda；
  同一个二进制也可以用 `--backend wgpu|cpu|ndarray` 运行。
- 命令：`target/release/examples/separate_all --backend cuda --output NO_TRACK/runs/examples/full-song-cuda`。
- 预热：正式运行前先跑过一次 30 秒片段（`--start 30 --duration 30`，用时 619.8 秒），为本示例的缓存作用域
  `ptx-sm61/separate_all` 生成 PTX 和 autotune 结果。下表不包含这部分冷启动代价，所以首块与稳态接近。

| 模型 | 块数 | 加载 | 模型 | 每块（首次 / 稳态） | 其它 | 总耗时 | RTF | 运行后显存占用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| leap-xe-voc | 30 | 1.66 s | 855.04 s | 28.57 / 28.50 s | 6.82 s | 863.52 s | 3.111 | 4556 MiB |
| deux | 45 | 5.42 s | 525.84 s | 11.89 / 11.68 s | 9.51 s | 540.77 s | 1.948 | 4622 MiB |
| hyperace-v2-voc | 58 | 1.81 s | 1040.74 s | 17.98 / 17.94 s | 11.86 s | 1054.41 s | 3.799 | 5522 MiB |
| hyperace-v2-inst | 58 | 1.80 s | 1040.88 s | 17.94 / 17.95 s | 12.07 s | 1054.74 s | 3.800 | 5522 MiB |
| UVR_MDXNET_9482 | 49 | 0.23 s | 19.71 s | 0.41 / 0.40 s | 5.25 s | 25.20 s | 0.091 | 5522 MiB |
| UVR_MDXNET_KARA | 49 | 0.22 s | 19.71 s | 0.40 / 0.40 s | 5.23 s | 25.16 s | 0.091 | 5522 MiB |
| UVR_MDXNET_KARA_2 | 48 | 0.41 s | 29.83 s | 0.62 / 0.62 s | 4.75 s | 34.99 s | 0.126 | 5522 MiB |
| UVR-MDX-NET-Inst_HQ_2 | 49 | 0.51 s | 46.88 s | 0.96 / 0.96 s | 5.57 s | 52.96 s | 0.191 | 5522 MiB |

“其它”等于总耗时减去加载和模型时间，包括解码、重采样、STFT / iSTFT、重叠相加和写文件。RTF 为总耗时除以音频时长，
大于 1 表示比实时慢。

- 8 个模型合计 3651.8 秒（约 61 分钟），进程峰值 RSS 为 2421 MiB。四个 RoFormer 类模型占其中的 96.2%。
- 稳态每块耗时与上一节的原生块测量一致（Leap 28.58 秒，HyperACE 17.94 秒）。全曲运行中最慢的稳态块比中位数
  最多慢 0.4%，长时间运行没有出现变慢。
- 显存一列是每个模型结束后 `cuMemGetInfo` 的 total − free，包含 CUDA 上下文和 CubeCL 内存池。内存池在进程内
  复用，模型释放后也不归还驱动，所以这一列只增不减，不能当作单个模型的峰值。
- 输出检查：所有 stem 都是有限值。不同模型的全曲人声两两比较，SDR 为 12.4–22.7 dB：RoFormer 类模型之间为
  18.2–22.7 dB，涉及 MDX 的组合为 12.4–15.1 dB。这只说明各模型给出的分离结果彼此一致，不是对真值的评测；
  逐模型的数值正确性见“数值一致性”一节的片段级参考比对。

明细（含每块耗时与两两 SDR）见 [full-song-cuda.json](reports/full-song-cuda.json)。

迁移包另有第二轮手写内核的整曲结果（同一首歌、8 个模型均完成）：总计 1189.49 秒（19.82 分钟），
峰值 RSS 2328 MiB。各模型总耗时如下；这是远端记录，本地 macOS 合入时未执行 CUDA。

| 模型 | 第二轮总耗时 | RTF |
|---|---:|---:|
| Leap Xe voc | 245.23 s | 0.883 |
| Deux | 155.76 s | 0.561 |
| HyperACE v2 voc | 355.51 s | 1.281 |
| HyperACE v2 inst | 355.80 s | 1.282 |
| MDX 9482 | 13.79 s | 0.050 |
| MDX KARA | 13.77 s | 0.050 |
| MDX KARA 2 | 19.85 s | 0.072 |
| MDX Inst HQ 2 | 29.71 s | 0.107 |

原始汇总的脱敏副本与来源摘要见 [cuda-kernels.json](reports/cuda-kernels.json) 的 `full_song` / `provenance`。

## profile 记录

Nsight Systems 2025.5.2（`-t cuda`）记录真实 GPU 内核时长，不做逐内核同步；3 秒片段，缓存已预热，单块。
二进制为 `fa6efbf`，即 HyperACE GEMM 卷积之前；两行 HyperACE 里的“直接卷积”就是该优化的依据：

| 场景 | 模型调用 | 内核数 | GPU 内核时间 | 占比 |
|---|---:|---:|---:|---|
| Leap 短块 | 2.69 s | 1216 | 2.58 s | MatMul 61.7%、融合逐元素 11.7%、拷贝 10.8%、二元/一元 9.9%、归约 4.6% |
| Deux 短块 | 1.81 s | 2200 | 1.63 s | MatMul 63.9%、融合逐元素 11.3%、二元/一元 9.5%、拷贝 9.4%、归约 4.4% |
| HyperACE voc 短块 | 2.08 s | 2513 | 1.80 s | MatMul 49.6%、直接卷积 13.9%、融合逐元素 11.2%、二元/一元 9.4%、拷贝 7.6%、归约 6.5% |
| Leap 原生块 | 28.75 s | 3134 | 28.60 s | MatMul 54.9%、二元/一元 15.1%、融合逐元素 14.7%、归约 8.0%、拷贝 6.3% |
| HyperACE voc 原生块 | 19.98 s | 3783 | 19.76 s | MatMul 48.1%、二元/一元 15.3%、融合逐元素 10.7%、归约 10.2%、直接卷积 9.4% |
| MDX KARA 2 | 0.67 s | 330 | 0.62 s | MatMul 61.2%、patch 拷贝 23.0%、融合逐元素 7.0%、选择/切片 5.2% |
| MDX Inst HQ 2 | 0.99 s | 330 | 0.95 s | MatMul 61.6%、patch 拷贝 22.4%、融合逐元素 6.9%、选择/切片 5.7% |
| MDX KARA | 0.47 s | 306 | 0.43 s | MatMul 62.0%、patch 拷贝 21.9%、融合逐元素 7.5%、选择/切片 5.1% |

- GPU 已基本饱和（内核时间约为模型调用的 87–99%），瓶颈在内核本身而不是启动或同步。
- MatMul 占 48–64%。Pascal 上 CubeCL 只有 unit / gemv 矩阵乘候选；Leap 短块的 MatMul 约 1.59 秒，
  按约 1.85 TFLOP 的计算量折合约 1.1–1.2 TFLOPS。同一 GPU 上 cuBLAS（PyTorch 2.5.1+cu121，关闭 TF32）完成同一组 Transformer GEMM
  约 0.56 秒（3.6–4.4 TFLOPS 的投影、1.2–1.9 TFLOPS 的注意力 batched GEMM）。
- 原生长块中逐元素、二元与归约合计 36–38%，主要是对大 score 张量做 softmax 时的多次显存往返。
- 每个进程的 `cuModuleLoadData`（驱动把缓存的 PTX JIT 成机器码）为 0.3–5.6 秒，基本与 GPU 执行重叠。

原始数据见 [cuda-profile.json](reports/cuda-profile.json)，GEMM 上限测量见 [cuda-ablation.json](reports/cuda-ablation.json)。

## 试验后未采用

| 方案 | 结果 | 处理 |
|---|---|---|
| Burn 融合 attention（autotune 选 flash-unit / fallback），按组分块调用 | Leap 短块 2.82 s 对 2.63 s，原生块 35.1 s 对 28.6 s，输出一致 | 未保留 |
| `--max-score-mib` 1024 / 2048 | 8 GB 显存不足，修复前静默出错 | 保持 512，并加设备故障检查 |
| MDX 用 Burn conv2d | 1.71 s 对 GEMM 0.65 s | auto 继续选 GEMM |
| attention 内核用 fast-math `__expf` | 90×1722 token 0.2099 → 0.2058 s（2%） | 精度变差、收益小，未保留 |
| attention 在内核里做 RoPE | 每个 query 块重新旋转全部 key，寄存器 80 → 103 | 移到 GEMM 收尾，每个元素只旋转一次 |
| attention score 循环全展开 | 0.200 → 0.399 s（寄存器溢出） | 未保留 |
| attention PV 循环展开 4 | 0.2005 → 0.1965 s（2%，在噪声内） | 未保留 |
| 自定义内核在 WGPU 上分派 | Deux 片段输出 SDR −5 dB | 只在 CUDA 分派 |

## 边界

- 只验证了一块 Pascal GPU（无 Tensor Core）。手写内核是 FP32 SIMT 实现，块大小与寄存器预算按 sm_61（48 KiB
  共享内存、64K 寄存器 / SM）调整；其它架构上需要重新测速并重跑参考比对，也未利用 Volta 以后的 Tensor Core。
  本机 GPU 停在 1113 MHz 应用时钟，未调整。
- 手写内核只在 CUDA 分派；CPU / WGPU 仍是原 Burn 路径。attention 内核只处理 head_dim 64，GEMM 要求
  `k % 8 == 0`、`m % 128 == 0`，卷积要求 `c·kh·kw % 8 == 0`，其余形状自动回退。
- PTX 缓存与 CubeCL 版本、算力绑定；PTX 由驱动在每个进程 JIT。降级驱动后若加载失败，删除对应 `ptx-sm*` 目录即可。
- 设备故障检查依赖 CubeCL 0.10 的设备线程命名；升级 Burn / CubeCL 时需复核。没有 profiler 级显存峰值。
- 本机 WGPU（NVIDIA Vulkan）对 Deux / HyperACE inst 比对失败，偶发在进程退出时驱动段错误；未定位，属已有问题。
- 速度数字来自单机、单首歌的固定片段，开发机同时有其它进程；中位数降低了但没有消除噪声。

## 下一步（按优先级）

1. attention 内核：原生块里仍占 GPU 时间的约一半（2.7 TFLOPS，约峰值的 48%）。拆解实验表明 score 循环、
   PV 循环、其余（装载、softmax、同步）大致各占 42% / 39% / 19%，循环受共享内存带宽限制；可试更大的寄存器块
   （需要压缩 48 KiB 内的共享内存布局）或在有 Tensor Core 的 GPU 上用 mma（不同精度配置，需单独 parity）。
2. HyperACE / MDX23C 的 InstanceNorm 已换成手写两遍归约（2026-10-03，MDX23C 单次 forward 热运行 1.537 s → 1.465 s）。深度可分离卷积仍用 Burn conv2d。逐元素 GELU 还没有并进这次归约。
3. 时间轴 / 频率轴之间的转置拷贝（Leap 约 0.18 s / 块）：让 GEMM 直接按跨步读写。
4. 加载：SHA-256 在无 SHA 扩展的 CPU 上是 Deux 加载的下限，可考虑按文件身份缓存已验证摘要（需要权衡完整性语义）。
5. WGPU / NVIDIA Vulkan 上 Deux 与 HyperACE inst 的数值偏差：逐层导出中间张量与 CPU 对比定位算子。

## 复现

```bash
PATH=/usr/local/cuda/bin:$PATH cargo build --release --locked --features cuda --examples
cargo test --release --locked --features cuda --test cuda_contracts --test cuda_device_failure
ANCHA_BACKENDS=cuda bash scripts/parity-matrix.sh            # 需 NO_TRACK/.venv-parity 与 NO_TRACK/reference
ANCHA_BACKENDS="cuda wgpu cpu" bash scripts/benchmark-backends.sh
target/release/examples/separate_all --backend cuda --start 30 --duration 30   # 预热缓存
target/release/examples/separate_all --backend cuda                            # 整首歌、全部模型
nsys profile -t cuda -o leap target/release/ancha separate NO_TRACK/runs/clip-3s.wav \
  --model NO_TRACK/models/leap-xe-voc --backend cuda --chunk-samples 132300 --overlap 1 --output /tmp/leap
# 手写内核：算子级对比（groups tokens [次数] / 行数）与整模型消融。
cargo run --release -p ancha-models --features cuda --example fused_attention -- 90 1722
cargo run --release -p ancha-models --features cuda --example fused_linear -- 154980
target/release/ancha separate NO_TRACK/runs/clip-3s.wav --model NO_TRACK/models/leap-xe-voc \
  --backend cuda --attention-kernel tiled --gemm-kernel burn --output /tmp/leap-round1
```

本轮的 nsys 汇总、预热与实验脚本保留在 `NO_TRACK/speed`（`cuda_profile.sh`、`nsys_summary.py`、
`gemm_ceiling.py`、`cuda_warm.sh`）。生成的 CUDA 源码可用 `CUBECL_DEBUG_LOG=/tmp/cubecl.log` 导出，
再用 `nvcc -arch=sm_61 -cubin --resource-usage` 查看寄存器与溢出；nsys 导出的 `registersPerThread` 是实际值。
