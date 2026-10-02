use anyhow::{Result, ensure};
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
pub type SpectrumComplex = Complex32;
use std::sync::Arc;

/// Contiguous frame-major spectrum; channel packing is handled by the runtime.
pub struct Spectrum {
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<Complex32>,
}

/// PyTorch center=true, periodic Hann, unnormalized FFT, reflect padding.
/// FFT plans and scratch buffers are reused across all chunks and channels.
pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    frequency: Vec<Complex32>,
    forward_scratch: Vec<Complex32>,
    inverse_scratch: Vec<Complex32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Result<Self> {
        ensure!(
            n_fft >= 2 && n_fft.is_multiple_of(2) && hop > 0 && hop <= n_fft / 2,
            "require even FFT >= 2 and hop in 1..=FFT/2"
        );
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        Ok(Self {
            n_fft,
            hop,
            window: (0..n_fft)
                .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n_fft as f32).cos())
                .collect(),
            time: forward.make_input_vec(),
            frequency: forward.make_output_vec(),
            forward_scratch: forward.make_scratch_vec(),
            inverse_scratch: inverse.make_scratch_vec(),
            forward,
            inverse,
        })
    }

    pub fn forward(&mut self, signal: &[f32]) -> Result<Spectrum> {
        ensure!(
            signal.len() > self.n_fft / 2,
            "reflect STFT requires input longer than FFT/2"
        );
        ensure!(
            signal.iter().all(|x| x.is_finite()),
            "non-finite STFT input"
        );
        let frames = signal.len() / self.hop + 1;
        let bins = self.n_fft / 2 + 1;
        let mut data = Vec::with_capacity(frames * bins);
        for t in 0..frames {
            for i in 0..self.n_fft {
                let index = t * self.hop + i;
                let reflected =
                    reflect_index(index as isize - self.n_fft as isize / 2, signal.len());
                self.time[i] = signal[reflected] * self.window[i];
            }
            self.forward.process_with_scratch(
                &mut self.time,
                &mut self.frequency,
                &mut self.forward_scratch,
            )?;
            data.extend_from_slice(&self.frequency);
        }
        Ok(Spectrum { frames, bins, data })
    }

    /// Explicit length; samples unsupported by the last frame are rejected.
    pub fn inverse(
        &mut self,
        spectrum: &Spectrum,
        length: usize,
        zero_dc: bool,
    ) -> Result<Vec<f32>> {
        ensure!(
            spectrum.bins == self.n_fft / 2 + 1
                && spectrum.frames > 0
                && spectrum.data.len() == spectrum.frames * spectrum.bins,
            "invalid spectrum shape"
        );
        let total = (spectrum.frames - 1) * self.hop + self.n_fft;
        let pad = self.n_fft / 2;
        ensure!(
            length + pad <= total,
            "requested iSTFT length exceeds frame support"
        );
        let mut output = vec![0.0f32; total];
        let mut denominator = vec![0.0f32; total];
        let scale = 1.0 / self.n_fft as f32;
        for t in 0..spectrum.frames {
            self.frequency
                .copy_from_slice(&spectrum.data[t * spectrum.bins..(t + 1) * spectrum.bins]);
            self.frequency[0].im = 0.0;
            self.frequency[spectrum.bins - 1].im = 0.0;
            if zero_dc {
                self.frequency[0].re = 0.0;
            }
            self.inverse.process_with_scratch(
                &mut self.frequency,
                &mut self.time,
                &mut self.inverse_scratch,
            )?;
            let start = t * self.hop;
            for (i, &w) in self.window.iter().enumerate() {
                output[start + i] += self.time[i] * scale * w;
                denominator[start + i] += w * w;
            }
        }
        let mut result = Vec::with_capacity(length);
        for i in pad..pad + length {
            ensure!(denominator[i] > 1e-11, "iSTFT has an uncovered sample");
            result.push(output[i] / denominator[i]);
        }
        ensure!(
            result.iter().all(|x| x.is_finite()),
            "non-finite iSTFT output"
        );
        Ok(result)
    }
}

/// Reflect padding excludes the edge; repeated reflection also handles short chunk borders.
pub fn reflect_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let i = index.rem_euclid(period);
    if i < length as isize {
        i as usize
    } else {
        (period - i) as usize
    }
}
