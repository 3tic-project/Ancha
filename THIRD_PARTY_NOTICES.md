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
