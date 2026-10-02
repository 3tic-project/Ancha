# 模块与数值契约

根包 `ancha` 同时提供 CLI 和阻塞 Rust SDK。`src/runtime.rs` 编排 decode → resample →
chunk → STFT → RoFormer → iSTFT → OLA → residual → WAV / run.json。

| 位置 | 职责 |
|---|---|
| `crates/ancha-audio` | Symphonia 解码、指定片段、Rubato sinc 重采样、RealFFT、分块和 OLA |
| `crates/ancha-models` | 版本化 manifest、严格权重加载、BS/Mel/HyperACE forward、经典 MDX ONNX 图、转换工具 |
| `crates/ancha-kernels` | 原实验包的标量数学参考、online softmax、融合与缓存准入测试 |
| `src` | 后端选择、资源限额、任务取消、CLI、运行报告、性能消融 |
| `tests` | 运行时生成的原创微型权重与音频，覆盖完整分离路径 |
| `scripts` | 开发期下载／Python 独立 parity；运行时不调用这些脚本 |
| `configs` | 不含权重的模型配置，Mel 索引在开发期导出后固定 |
| `NO_TRACK` | 参考工程、原始歌曲、模型、Python 环境、产物与本机日志；Git 排除 |

## 模型兼容

模型包为 `manifest.json` + `model.safetensors`。schema 1 只接受 FP32、44100 Hz、双声道、
FFT 2048、每轴单层 Transformer、标准 residual 路径。未知 JSON 字段、键、形状、dtype、
非有限权重、未用张量和 SHA256 不匹配都返回错误。直接依据 PyTorch 键加载，Linear 权重
从 `[out,in]` 转成 Burn `[in,out]`，没有随机初始化或部分加载。

BS-RoFormer 不在每轴末尾添加 norm，完整主干结束执行 `final_norm`。
Mel-Band 在每轴末尾执行 norm，没有同样的 `final_norm`。
BS 的 MLP depth 是 Linear 总层数；Mel 的 depth 是隐藏层数。两者不可共享硬编码层数。

源 RMSNorm 为 `sqrt(dim) * gamma * x / max(sqrt(sum(x*x)), 1e-12)`。
不使用 Burn 内置的 `sqrt(mean(x*x)+eps)` 公式。RoPE 旋转相邻二元组；checkpoint frequencies
参与加载。Attention 是 non-causal、全局 K/V，之后应用每头 sigmoid gate。
MLP 使用 Tanh 和最终 GLU，FFN 使用精确 GELU。

Mel gather 保留重复频点，输出 mask 用 scatter-add 聚合，再除以每频点覆盖数量。
双输出模型的两个原生头均保留，单输出模型才计算另一轨残差。
HyperACE v2 使用独立 SegmModel adapter，把空间预测与 per-band MLP mask 相加；完整参数必须被消费。
它不会被当作普通 BS 模型，固定 source/DSP 与后续验收见 [新增适配](adapters.md)。

### 推理期等价变换

加载时只做代数等价的折叠，checkpoint 的严格消费检查不变：

- RMSNorm 的 `sqrt(dim)·gamma` 只喂给其后的投影（band split、q/k/v/gate、FFN 输入），
  折进这些投影的输入行；运行期只保留 `x / max(||x||, 1e-12)`。final norm 和 Mel 轴末 norm
  另有其他读者，保留独立缩放。
- `to_qkv` 拆成 q/k/v 三个连续输出；1/sqrt(head_dim) 乘进 query 的 RoPE cos/sin 表
  （RoPE 线性），head_dim=64 时为精确的 2 的幂。RoPE 用 `x·cos + swap_pairs(x)·sin±`，
  成对符号预置在 sin 表中，每次 forward 每轴只构建一次表。
- 相同输入宽度的频带合成一组，band split 与 mask 末层按组做批量 GEMM（feature-major，
  bias 形状 `[n,out,1]`）；Mel 的非相邻同宽频带通过一次 gather 恢复原频带顺序。
- 网络输出复数 mask `[stems, rows, re/im, frames]`，复数乘回原频谱在主机 iSTFT 前完成。

这些变换只改变浮点舍入顺序，真实权重的 PyTorch 波形比对见 [速度优化记录](speed-optimization.md)。

## DSP、分块与输出

输入限制为 WAV / FLAC / MP3，mono 复制为 stereo；多于两个声道明确报错。
Symphonia gapless 解码跳过封面等非音轨包；切片按实际解码采样点计数，不以 MP3 包边界截取。
不同采样率使用 sinc 插值并固定到四舍五入后的输出样本数，立体声共用相同采样时序。
脉冲测试覆盖绝对位置和声道对齐。

STFT 使用 periodic Hann、center=true、反射 padding 和未归一化 FFT；iSTFT 用窗平方归一化，
显式恢复 chunk 长度，支持非 hop 整数倍尾部。按照模型的 `zero_dc` 在逆变换前清除 DC。
RealFFT 的计划、频谱、时间和 scratch buffers 在通道、chunk 间复用。

默认采用 manifest 的 chunk 与 overlap divisor，step=`chunk_samples / overlap`。
长输入两侧添加 `chunk-step` 反射 border，再在 OLA 后裁掉。
尾块超过半块时反射补齐，否则补零。Ancha 的正权重线性淡入淡出避免 overlap=1 时分母为零；
它不是 MSST 含首尾零权重窗口的逐采样复制。整轨一致性需使用相同 Ancha 分块规则作为参照。
`--chunk-samples` / `--overlap` 会标记 `custom-context`，不是等质量加速。

输出为原增益的 float32 WAV；不做每轨归一化和削波。单输出模型的 predicted+residual
按浮点误差重建原 PCM；这个指标只证明残差语义，不能衡量分离质量。没有干净人声／伴奏真值
就不报告 SDR。输出只发布到新的目录，WAV 与 run.json 完整后原子 rename；失败和取消不发布产物。

## 内存、性能与取消

GPU 只驻留权重、当前 chunk/batch 和 forward 临时张量；不驻留整首歌曲。
主机保留解码 PCM 与输出 accumulator，内存仍随时长增长，通过 `--max-seconds` 限制采样数。
schema 1 的分离不是滚动磁盘流式实现。64 位平台默认每通道 158760000 samples；
三个双声道 PCM 缓冲本身可接近 3.8 GB，另需 chunk、权重、临时 buffer 和双输出工作内存。

attention 的每次 scores 张量为 `group_tile*heads*query_tile*tokens*4` 字节；时间轴
groups=bands、tokens=frames，频率轴相反。未给 `--query-tile` / `--group-tile` 时，
按 `--max-score-mib`（默认 512）选择不超过预算的最大精确分块：先保持整段 query，再尽量合并组。
分块只改变浮点调度，softmax 始终覆盖全长 K/V，未修改双向注意力上下文。
CPU 多 worker 并发时预算平分给各 worker，并发 scores 总量仍不超过该限额。
`--max-score-mib` 不是总显存上限；Burn 队列、QKV 和 concat 也需要内存。
没有量化、近似窗口或不合法的深层跨 chunk KV 缓存。

CPU 默认使用 Burn Flex 后端（gemm 矩阵乘、im2col 卷积、零拷贝 strided view）；
`--backend ndarray` 保留旧 NdArray 供消融。Flex 的逐元素算子是单线程，因此 RoFormer
每个轴向 Transformer 把相互独立的序列组（时间轴的频带、频率轴的帧）切给 `--host-threads`
个主机线程，默认等于逻辑核数；每组算术完全相同，WGPU 固定为 1。

模型计时包含 CPU→GPU 传输与最终 readback 同步，加载计时另列；`model_call_seconds`
逐次记录，第一项还包含 WGPU 内核编译和 autotune。
RTF=`包含加载与WAV写出的总墙钟时间 / 片段音频时间`，越小越快。
批量投影是本机更快的默认布局；独立行合并成单次大 GEMM 的实验路径保持数值一致，
但在 RX 580 上实测变慢，只有显式 `--flatten-linear` 才启用。
独立算子 `bench` 与完整音频分离耗时分开，避免将合成 GEMM 提速套到整模型。

SDK 接受 AtomicBool 取消标志和 chunk 进度回调。Ctrl+C 设置取消标志；在 chunk、
Transformer 层、mask head 边界检查，已经提交的 GPU kernel 可能需要先完成。

## 新增模型适配

`Family::HyperaceV2` 注册独立 SegmModel，在普通 BS per-band mask 上叠加空间分支；
所有新张量必须严格加载，resolved DSP 为作者的 zero_dc=false / 960000 / overlap=4。
空间模块在 `ancha-models::spatial`，完整网络在 `hyperace.rs`。

`ancha-models::mdx` 从 ONNX 直接建立 classic MDX 图调度，运行期权重常驻、按消费者释放中间句柄。
`ancha::mdx_runtime` 独立保持 UVR 的 FFT6144/5120、hop1024、F/T、complex packing、低频清零、
trim、padding、OLA、compensate 和任务标签；不滥用 schema1 RoFormer 的固定 FFT2048 配置。
基于 SHA256 的注册避免以文件名猜测模型。参数与加速边界见 [适配文档](adapters.md)。

CPU 默认 Flex 后端自带 SIMD 与 Rayon 并行 gemm/im2col，线程数可用进程启动前的
`RAYON_NUM_THREADS` 控制。可选 `cpu-opt` 只影响 `--backend ndarray`（Burn SIMD 与 NdArray 多线程），
macOS `accelerate` 为 NdArray 提供 BLAS；报告保存这些环境设置，但它们不是对实际并发线程的采样。
WGPU 启用 Burn autotune，固定 FP32，batch 默认 1。CubeCL 在没有 cooperative-matrix 的 GPU
（如 RX 580）上只能用直接卷积，MDX 因此默认把非分组卷积改写为 patch gather + 一次 GEMM；
CPU 仍用 Flex 原生卷积。`--conv-strategy gemm|backend` 可做同二进制消融。
