//! Stereo complex packing shared by MDX families. Low-bin suppression is explicit.
use ancha_audio::dsp::{Spectrum, SpectrumComplex, Stft};
use anyhow::{Result, ensure};

pub fn pack(
    stft: &mut Stft,
    planes: &[Vec<f32>],
    bins: usize,
    zero_low_bins: usize,
) -> Result<(Vec<f32>, usize)> {
    ensure!(
        planes.len() == 2 && planes[0].len() == planes[1].len(),
        "spectral packing requires equal stereo planes"
    );
    let frames = planes[0].len() / stft.hop + 1;
    let mut packed = Vec::with_capacity(4 * bins * frames);
    for plane in planes {
        let spectrum = stft.forward(plane)?;
        ensure!(bins > 0 && bins <= spectrum.bins, "bins exceed FFT");
        for ri in 0..2 {
            for f in 0..bins {
                for t in 0..frames {
                    let z = spectrum.data[t * spectrum.bins + f];
                    packed.push(if f < zero_low_bins {
                        0.
                    } else if ri == 0 {
                        z.re
                    } else {
                        z.im
                    });
                }
            }
        }
    }
    Ok((packed, frames))
}

pub fn unpack(
    stft: &mut Stft,
    packed: &[f32],
    bins: usize,
    frames: usize,
    length: usize,
) -> Result<Vec<Vec<f32>>> {
    let full = stft.n_fft / 2 + 1;
    ensure!(
        bins > 0
            && bins <= full
            && packed.len() == 4 * bins * frames
            && packed.iter().all(|v| v.is_finite()),
        "invalid spectral output"
    );
    let mut planes = Vec::new();
    for channel in 0..2 {
        let mut data = vec![SpectrumComplex::new(0., 0.); full * frames];
        for t in 0..frames {
            for f in 0..bins {
                data[t * full + f] = SpectrumComplex::new(
                    packed[(channel * 2 * bins + f) * frames + t],
                    packed[((channel * 2 + 1) * bins + f) * frames + t],
                );
            }
        }
        planes.push(stft.inverse(
            &Spectrum {
                frames,
                bins: full,
                data,
            },
            length,
            false,
        )?);
    }
    Ok(planes)
}
