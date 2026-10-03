# Mel Karaoke 与 MDX23C HQ2

两个 checkpoint 均已实现原生 Rust CLI / SDK 适配，运行时无需 Python、PyTorch 或 ONNX Runtime。
权重、开发参考、音频和完整日志只保存在 NO_TRACK；Git 保存代码、配置、合成测试与数值报告。

## 模型与任务

来源为 [Derur/UVR-models](https://huggingface.co/Derur/UVR-models/tree/f3bb9a312519f4404dde996ef1054ec30353c46f)，
固定修订 `f3bb9a312519f4404dde996ef1054ec30353c46f`，不随 main 自动更新。

| 项目 | Mel Karaoke aufr33 / viperx | MDX23C-8KFFT-InstVoc HQ2 |
|---|---|---|
| 本地包 / preset | `mel-karaoke-aufr33-viperx` | `mdx23c-inst-voc-hq2` |
| 原生预测 | `lead_vocals.wav` | `vocals.wav`、`instrumental.wav` |
| 残差 | `karaoke_mix.wav` | 无，两个输出都是 predicted |
| FFT / hop / 采样率 | 2048 / 441 / 44100 | 8192 / 1024 / 44100 |
| 原生 chunk / overlap divisor | 352800 / 4 | 261120 / 8 |
| 严格加载张量 | 684 | 319 |
| 结构 | dim384、6 层、60 Mel bands、8×64 attention | 4 subbands、5 scales、每级 2 blocks、128 起始 channels |

Karaoke 模型预测主唱；残差保留其他成分，命名为 karaoke_mix，不能视为不含和声的纯伴奏。
任务标签依据 [维护者模型映射](https://github.com/nomadkaraoke/python-audio-separator/blob/main/docs/deton24-model-mapping-and-ensemble-guide.md)。
模型名称中的 SDR 数字不代表本次测试的质量指标。本次没有干净源真值，只验证实现一致性与运行速度。

## 下载、转换、运行

```bash
cargo build --release --locked --features convert,accelerate  # macOS
bash scripts/download-derur-models.sh

target/release/ancha convert \
  NO_TRACK/models/derur-download/mel_band_roformer_karaoke_aufr33_viperx_sdr_10/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt \
  --preset mel-karaoke-aufr33-viperx --output NO_TRACK/models/mel-karaoke-aufr33-viperx
target/release/ancha convert \
  NO_TRACK/models/derur-download/MDX23C-8KFFT-InstVoc_HQ_2/MDX23C-8KFFT-InstVoc_HQ_2.ckpt \
  --preset mdx23c-inst-voc-hq2 --output NO_TRACK/models/mdx23c-inst-voc-hq2

target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/mel-karaoke-aufr33-viperx --backend wgpu \
  --start 30 --duration 30 --output NO_TRACK/runs/mel-karaoke-30s
target/release/ancha separate 'NO_TRACK/test_file/ReoNa - Amore.mp3' \
  --model NO_TRACK/models/mdx23c-inst-voc-hq2 --backend wgpu \
  --start 30 --duration 30 --output NO_TRACK/runs/mdx23c-30s
```

Linux / Windows 去掉 `accelerate`；NVIDIA 加 `--features cuda` 并选择 `--backend cuda`。
本机无 NVIDIA GPU，这两个新模型的 CUDA 仅做编译检查，不能套用旧模型的 P4 验收结果。
`inspect <模型包>` 可查看完整契约；输出目录必须尚不存在。

下载脚本优先用 `hf download`，也支持 curl，并核验 checkpoint 与 YAML 的固定 SHA256。
转换 preset 再核验 checkpoint，Rust 转 F32 Safetensors 后构造完整 CPU 模型，拒绝未消费键、
缺键、形状错误、非有限值和摘要不匹配。模型包带源 checkpoint 摘要与参考实现修订。

| 原始文件 | SHA256 |
|---|---|
| Mel checkpoint | `1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813` |
| Mel YAML | `b35077d94861f068097cce1a5e54633c055e7dcc2613eade4e4dc7c7c9c3f48b` |
| MDX23C checkpoint | `7d960d8e40a458120412c1bd807e013d2dbca7b959cc9da2bbcb0eb203d1daea` |
| MDX23C YAML | `451765e869b78dcb9ca9188a74da31f581b7254ff0e8b532aa76b974148de947` |

## DSP 与等价加速

Mel Karaoke 复用已验收的 Mel RoFormer，并保留该 checkpoint 的 dim384 / depth6 / 单头配置。
同宽 band 分组、RMSNorm 系数折叠、RoPE 缩放、自动完整注意力分块和 CPU 分组并行均复用现有路径。
`--linear-layout batched|flattened` 可测布局差异，不改变全局 K/V 上下文。

MDX23C 是独立的 TFC/TDF v3 网络，参考
[UVR 固定源码](https://github.com/Anjok07/ultimatevocalremovergui/blob/5517e0cf0d1acd16a1618eeedec596957523f9e1/lib_v5/tfc_tdf_v3.py)。
输入为 `[L.re,L.im,R.re,R.im]`，保留 DC 与前 3 个频点，去掉 Nyquist，按连续频段折入 channels。
TDF 沿频率轴，InstanceNorm 使用 biased variance / eps=1e-5；HyperACE 仍使用 eps=1e-8。
输出是双头复数频谱，不是乘回输入的 mask。

重叠规则直接对齐 UVR MDXC demix：step=`chunk//overlap`，border=`chunk-step`，两侧零上下文，
pad=`step-((length-chunk)%step)` 使用 Python 非负模，完整矩形窗口累加后除以常数 overlap。
本机 3 秒原生上下文需要 12 次 forward；30 秒需要 48 次，短音频也不能省掉参与结果的边界窗口。
`--mdx-no-optimize` 保留 CUDA 基线 TDF 路径和无用尾块，供同二进制消融。

MDX23C 的等价变换包括：CUDA 上 TDF 的独立行合并并使用已有 FP32 自定义 GEMM；
2×2 stride2 非重叠反卷积改成一次 GEMM 后按相位重排；卷积复用 GEMM 路径，CUDA 可复用隐式卷积内核。
CPU 默认后端卷积，WGPU / CUDA 默认 GEMM；`--conv-strategy gemm|backend` 可比较。
只跳过完全不与最终裁剪区相交的尾窗口。没有降低精度、频率截断、近似注意力或修改 checkpoint。

首轮通用 flattened TDF / 单轴 norm 在 RX 580 的 30 秒验收出现 max_abs≈0.00104，超过既定 0.001 门槛。
定位到第 23.47 秒对应窗口后，基线达到约 7e-7；只恢复双轴 norm 后仍约 0.00198，
因此失败定位到通用 flattened TDF GEMM。CPU 上这组变化仅约 0.8% 提升。
生产 CPU / WGPU 保留 Burn 原始批量 Linear 和双轴 norm；问题窗口最终 max_abs < 8e-7，
完整 30 秒原生重叠的两个头分别约 2.98e-7 / 3.58e-7，均通过原门槛。
CUDA 只有自定义 GEMM 支持形状才使用它，
否则也回到原 Linear。失败记录保留在报告中，不以宽松阈值作为通过依据。

`--duration` 只截输入。显式 `--overlap 1` 会改变上下文覆盖，报告为 custom-context。
MDX23C 的 `--chunk-samples` 必须为 `1024*(frames-1)`，frames 为 32 的倍数且在合法范围内；
例如 64512 对应 64 frames。缩短块或减少 overlap 的速度不能当作等质量提速。
WGPU 首次新形状会 autotune，冷启动与预热测量分开。patch gather 卷积会占用额外显存；
自动 attention 预算不适用于 MDX23C，也不是总显存上限。

## 测试与复现

```bash
cargo fmt --all --check
cargo test --workspace --all-targets --locked --features convert,accelerate
cargo clippy --workspace --all-targets --locked --features convert,accelerate -- -D warnings
CUDARC_CUDA_VERSION=12020 cargo check --workspace --all-targets --locked --features convert,cuda

bash scripts/fetch-reference.sh
bash scripts/fetch-extra-reference.sh
# Python 只用于开发比对，按 development.md 安装 NO_TRACK/.venv-parity。
ANCHA_BACKENDS="cpu wgpu" bash scripts/parity-derur.sh

# 等 PCM / 同上下文；每种先预热，交替顺序重复三对，记录冻结二进制摘要。
NO_TRACK/.venv-parity/bin/python scripts/benchmark-derur.py \
  --model NO_TRACK/models/mdx23c-inst-voc-hq2 --backend wgpu --ablation conv \
  --overlap 1 --output NO_TRACK/runs/mdx23c-wgpu-ablation
NO_TRACK/.venv-parity/bin/python scripts/benchmark-derur.py \
  --model NO_TRACK/models/mel-karaoke-aufr33-viperx --backend wgpu --ablation linear \
  --chunk-samples 132300 --overlap 1 --output NO_TRACK/runs/mel-karaoke-layout
```

新增 7 项合成契约涵盖独立 PyTorch 完整双尺度双头 golden、batch / 相位重排反卷积、
eps / biased variance、DC / complex packing、原生矩形 OLA 边界、严格加载与取消后不发布。
golden 使用原创微型权重公式，不包含发布模型权重或歌曲。旧 HyperACE / 经典 MDX 测试继续运行。
`scripts/make-mdx23c-fixture.py --output <文件>` 可用固定 UVR 源码重新生成 golden；
CPU golden 另要求相对 MSE < 1e-8，拒绝以接近零的输出混过绝对误差门槛。
开发脚本 `verify_mdx23c.py` 校验固定源代码与 checkpoint 摘要，直接提取 UVR 原始 demix 方法；
参考缓存按 PCM、配置、源文件、权重与 Torch 版本隔离并核验缓存 WAV 摘要。
max_abs < 1e-3、waveform SNR > 50 dB 为一致性门槛，不是分离 SDR。

2026-10-03，本机 Intel i5-12400 / 64 GB / RX 580 8 GB 的验收：

| 场景 | max_abs | waveform SNR |
|---|---:|---:|
| Mel，CPU，3 秒单块 | 4.55e-7 | 127.5 dB |
| Mel，WGPU，3 秒单块 | 4.50e-4 | 69.3 dB |
| MDX23C，CPU，3 秒原生 overlap8，双头 | ≤8.05e-7 | ≥129.1 dB |
| MDX23C，WGPU，3 秒原生 overlap8，双头 | ≤3.58e-7 | ≥137.6 dB |
| MDX23C，WGPU，30 秒原生 overlap8，双头 | ≤3.58e-7 | ≥137.9 dB |

Mel 的 30 秒原生上下文完成 21 个块，验证有限值、双声道长度与残差重建；重建 max_abs=5.96e-8。
它尚未做整段独立 PyTorch 分块比对。MDX23C 的 30 秒参考直接执行 UVR 原始 demix 的 48 个窗口。
完整修订、SHA、门槛、回退记录与 CUDA 编译边界见 [验收数值](reports/derur-verification.json)。

Rust SDK：Mel 使用 `ancha::backend::separate`；MDX23C 使用
`ancha::backend::separate_mdx23c` + `ancha::mdx23c_runtime::Options`，共用取消标志和进度回调。
`examples/separate` 可识别两个包，`separate_all --only mel-karaoke-aufr33-viperx,mdx23c-inst-voc-hq2`
可运行两个新模型。MDX23C 当前任务 batch=1，无 pitch shift、denoise 或 UVR GUI 后处理。
