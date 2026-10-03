# 来源与许可

Ancha 原创代码使用根目录的 MIT LICENSE。模型权重和测试歌曲不随 Git 仓库分发。

`crates/ancha-kernels` 的 attention、fusion、cache 和原始 7 项契约测试来自本工作区的
`NO_TRACK/acceleration-lab`（0.1.0，2026-10-02，MIT）。保留为独立标量参考；生产模型使用 Burn 张量与 GEMM。

`crates/ancha-models/src/convert.rs` 的 checkpoint 读取方案改编自
[dentimoer-official/eplyt](https://github.com/dentimoer-official/eplyt/tree/fbd7c64b34fe781095f6563ce576fb21828a4af6)，
`src/audio_process/weights/convert.rs`，MIT。原始 MIT 文本：

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

完整 forward 语义依据固定修订的
[MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/84b1eac0887756b4f1a9d7a1ff49105939749ed2)
重新实现。独立 Python 数值参考源码仅下载到 NO_TRACK，仓库只保存获取脚本与修订号。
Burn 0.21.0、CubeCL 0.10、RealFFT、Symphonia、Rubato 等依赖许可由各自项目规定，准确版本见 Cargo.lock。

Leap Xe 权重来源为
[pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e)。
固定模型修订未明确声明权重许可；代码的 MIT 不扩展至这些权重。
Deux 的模型卡标注 CC-BY-NC-4.0，模型包需保留作者来源与许可。NO_TRACK 中的歌曲仅用于用户授权的本地验证。

HyperACE 的 Rust 适配根据
[pcunwa/BS-Roformer-HyperACE 固定源码](https://huggingface.co/pcunwa/BS-Roformer-HyperACE/tree/5b1f8283125d5e4a3614d0e3635a636e09c84059)
重新实现，源码仅用于 NO_TRACK 中的独立参考。v2 voc/inst 的专用 bs_roformer.py 摘要相同；
不能因为其可经 MSST 入口运行就把该专用作者文件、checkpoint 归入 MSST 的 MIT 许可。
下载与转换脚本保留作者来源，权重许可状态仍为未明确声明，不随 Git 分发。

经典 MDX 的模型、SHA256、任务标签及 UVR 参数从用户提供的
`NO_TRACK/lightweight-separation-lab/model-manifest.json` 和 UVR 固定修订
[`5517e0cf0d1acd16a1618eeedec596957523f9e1`](https://github.com/Anjok07/ultimatevocalremovergui/tree/5517e0cf0d1acd16a1618eeedec596957523f9e1)
核对。图执行器和 DSP 是 Rust 重实现；独立参考脚本仅从已下载的 UVR 文件读取原始方法。
公开权重由 [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
单独提供，不因 UVR GUI 的代码许可而推定权重许可。
onnx-rs 0.1.2 为 MIT。默认 CPU 后端为 Burn Flex 0.21.0（MIT OR Apache-2.0，内部使用 gemm、
macerator 与 Rayon）；可选旧 CPU 路径由 Burn NdArray 的 SIMD/Rayon 实现提供，
准确依赖、版本及许可证见 Cargo.lock 和上游 crate 元数据。

Mel Karaoke aufr33 / viperx 与 MDX23C InstVoc HQ2 权重和 YAML 来自用户指定的
[Derur/UVR-models 固定修订](https://huggingface.co/Derur/UVR-models/tree/f3bb9a312519f4404dde996ef1054ec30353c46f)。
镜像未明确声明这两份权重的许可，代码 MIT 不扩展到权重；原始与转换权重仅保留在 NO_TRACK。
Mel forward 依据上述固定 MSST，MDX23C TFC/TDF v3 和矩形 OLA 依据上述固定 UVR 重实现。
独立参考源码不进入 Git；合成 golden 数据由原创微型权重生成，不包含真实权重或用户歌曲。
