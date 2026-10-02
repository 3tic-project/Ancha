//! Audio decoding, resampling, STFT and overlap-add.
pub mod chunk;
pub mod decode;
pub mod dsp;

use anyhow::{Result, ensure};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Audio {
    pub sample_rate: u32,
    pub planes: Vec<Vec<f32>>,
}

impl Audio {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.sample_rate > 0, "sample rate must be positive");
        ensure!(!self.planes.is_empty(), "no audio channels");
        let n = self.planes[0].len();
        ensure!(n > 0, "audio is empty");
        ensure!(
            self.planes.iter().all(|c| c.len() == n),
            "channel lengths differ"
        );
        ensure!(
            self.planes.iter().flatten().all(|x| x.is_finite()),
            "non-finite PCM"
        );
        Ok(())
    }

    pub fn stereo(mut self) -> Result<Self> {
        self.validate()?;
        match self.planes.len() {
            1 => self.planes.push(self.planes[0].clone()),
            2 => {}
            n => anyhow::bail!("{n} channels are unsupported; provide mono or stereo"),
        }
        Ok(self)
    }

    pub fn samples(&self) -> usize {
        self.planes.first().map_or(0, Vec::len)
    }
    pub fn seconds(&self) -> f64 {
        self.samples() as f64 / self.sample_rate as f64
    }
}

/// Float32 WAV, with no clipping or gain normalization.
pub fn write_wav(path: &Path, audio: &Audio) -> Result<()> {
    audio.validate()?;
    let spec = hound::WavSpec {
        channels: audio.planes.len().try_into()?,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for i in 0..audio.samples() {
        for channel in &audio.planes {
            writer.write_sample(channel[i])?;
        }
    }
    writer.finalize()?;
    Ok(())
}

/// Windowed-sinc resampling with an exact rounded output length.
pub fn resample(audio: Audio, target_rate: u32) -> Result<Audio> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    audio.validate()?;
    ensure!(target_rate > 0, "target sample rate must be positive");
    if audio.sample_rate == target_rate {
        return Ok(audio);
    }
    let ratio = target_rate as f64 / audio.sample_rate as f64;
    let length = (audio.samples() as f64 * ratio).round() as usize;
    ensure!(length > 0, "resampled audio is empty");
    let block = 1024;
    let mut rs = SincFixedIn::<f32>::new(
        ratio,
        1.0,
        SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        },
        block,
        audio.planes.len(),
    )?;
    // SincFixedIn 0.16 starts at -sinc_len/2; its initial interpolation center
    // already compensates the FIR lookahead. Removing output_delay() here
    // shifts the waveform early (58 samples for 48 kHz -> 44.1 kHz).
    // The impulse regression test verifies absolute sample alignment.
    let mut out = vec![Vec::with_capacity(length + block); audio.planes.len()];
    let mut offset = 0;
    while offset < audio.samples() || out[0].len() < length {
        let input: Vec<Vec<f32>> = audio
            .planes
            .iter()
            .map(|c| {
                (0..block)
                    .map(|i| c.get(offset + i).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        for (dst, chunk) in out.iter_mut().zip(rs.process(&input, None)?) {
            dst.extend(chunk);
        }
        offset += block;
    }
    for c in &mut out {
        c.truncate(length);
    }
    let result = Audio {
        sample_rate: target_rate,
        planes: out,
    };
    result.validate()?;
    Ok(result)
}
