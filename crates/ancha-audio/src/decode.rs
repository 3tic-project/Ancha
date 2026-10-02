use crate::Audio;
use anyhow::{Context, Result, bail, ensure};
use std::{fs::File, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error, formats::FormatOptions,
    io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub start_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub max_samples_per_channel: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            start_seconds: 0.0,
            duration_seconds: None,
            max_samples_per_channel: 44_100 * 60 * 60,
        }
    }
}

/// Decode only the requested slice. Cover-art packets and other tracks are skipped.
/// Packet PCM is consumed sequentially so clipping uses decoded sample indices.
pub fn decode(path: &Path, options: DecodeOptions) -> Result<Audio> {
    ensure!(
        options.start_seconds.is_finite() && options.start_seconds >= 0.0,
        "start must be finite and nonnegative"
    );
    if let Some(d) = options.duration_seconds {
        ensure!(
            d.is_finite() && d > 0.0,
            "duration must be finite and positive"
        );
    }
    let stream = MediaSourceStream::new(
        Box::new(File::open(path).with_context(|| format!("open audio {}", path.display()))?),
        Default::default(),
    );
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        stream,
        &FormatOptions {
            enable_gapless: true,
            ..Default::default()
        },
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format.default_track().context("no audio track")?;
    let id = track.id;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut sample_rate = 0;
    let mut planes = Vec::<Vec<f32>>::new();
    let mut position = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != id {
            continue;
        }
        let pcm = decoder.decode(&packet).context("decode packet")?;
        let spec = *pcm.spec();
        let channels = spec.channels.count();
        if planes.is_empty() {
            sample_rate = spec.rate;
            planes = vec![Vec::new(); channels];
        }
        ensure!(
            sample_rate == spec.rate && channels == planes.len(),
            "audio format changed mid-stream"
        );
        let start = (options.start_seconds * sample_rate as f64).round() as usize;
        let count = options
            .duration_seconds
            .map(|d| (d * sample_rate as f64).round() as usize);
        let end = count.map_or(usize::MAX, |n| start.saturating_add(n));
        let frames = pcm.frames();
        let mut buffer = SampleBuffer::<f32>::new(pcm.capacity() as u64, spec);
        buffer.copy_interleaved_ref(pcm);
        let from = start.saturating_sub(position).min(frames);
        let to = end.saturating_sub(position).min(frames);
        if to > from {
            ensure!(
                planes[0].len().saturating_add(to - from) <= options.max_samples_per_channel,
                "decoded PCM exceeds the host memory budget; use --duration or raise --max-seconds"
            );
            for frame in buffer.samples()[from * channels..to * channels].chunks_exact(channels) {
                for (dst, value) in planes.iter_mut().zip(frame) {
                    dst.push(*value);
                }
            }
        }
        position = position
            .checked_add(frames)
            .context("sample index overflow")?;
        if position >= end {
            break;
        }
    }
    if planes.is_empty() || planes[0].is_empty() {
        bail!("requested range contains no audio");
    }
    let audio = Audio {
        sample_rate,
        planes,
    };
    audio.validate()?;
    Ok(audio)
}
