use ancha_audio::{
    Audio,
    chunk::{Accumulator, ChunkPlan},
    decode::{DecodeOptions, decode},
    dsp::{Stft, reflect_index},
    resample, write_wav,
};

fn signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.071).sin() + 0.2 * (i as f32 * 0.31).cos()) * 0.4)
        .collect()
}

#[test]
fn stft_inverse_preserves_edges_odd_lengths_and_nonaligned_hops() {
    for hop in [441, 512] {
        let mut stft = Stft::new(2048, hop).unwrap();
        for n in [1025, 2048, 4103, 88201] {
            let mut input = signal(n);
            input[0] = 0.81;
            input[n - 1] = -0.72;
            let spec = stft.forward(&input).unwrap();
            assert_eq!(spec.frames, n / hop + 1);
            let output = stft.inverse(&spec, n, false).unwrap();
            assert_eq!(output.len(), n);
            let max = input
                .iter()
                .zip(output)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max < 2e-6, "hop={hop} n={n} max={max}");
        }
    }
}

#[test]
fn stft_matches_analytical_centered_periodic_hann_dc() {
    let mut stft = Stft::new(2048, 512).unwrap();
    let spectrum = stft.forward(&vec![1.0; 4096]).unwrap();
    assert!((spectrum.data[0].re - 1024.0).abs() < 1e-3);
    assert!((spectrum.data[1].re + 512.0).abs() < 1e-3);
    assert!(spectrum.data[2].norm() < 1e-3);
    let dc_filtered = stft.inverse(&spectrum, 4096, true).unwrap();
    assert!(dc_filtered.iter().all(|x| x.is_finite()));
    assert!(Stft::new(2047, 512).is_err());
    assert!(stft.forward(&[0.0; 1024]).is_err());
    assert!(stft.forward(&vec![f32::NAN; 2048]).is_err());
}

#[test]
fn reflect_padding_excludes_edge_and_handles_repeated_reflections() {
    let indices: Vec<_> = (-4..8).map(|i| reflect_index(i, 4)).collect();
    assert_eq!(indices, [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]);
    assert_eq!(reflect_index(-5, 1), 0);
}

#[test]
fn overlap_add_recovers_identity_at_every_boundary() {
    for n in [1, 31, 64, 65, 128, 303] {
        for overlap in [1, 2, 4] {
            let plan = ChunkPlan::new(64, overlap).unwrap();
            let input = signal(n);
            let mut acc = Accumulator::new(2, n).unwrap();
            for start in plan.starts(n) {
                let x: Vec<_> = (0..64)
                    .map(|i| input.get(start + i).copied().unwrap_or(0.))
                    .collect();
                acc.add(plan, start, &[x.clone(), x.iter().map(|v| -*v).collect()])
                    .unwrap();
            }
            let output = acc.finish().unwrap();
            assert!(
                input
                    .iter()
                    .zip(&output[0])
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            );
            assert!(
                input
                    .iter()
                    .zip(&output[1])
                    .all(|(a, b)| (a + b).abs() < 1e-6)
            );
        }
    }
    assert!(ChunkPlan::new(64, 0).is_err());
    assert!(Accumulator::new(2, 4).unwrap().finish().is_err());
}

#[test]
fn wav_clip_keeps_sample_count_channel_order_and_gain() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("synthetic.wav");
    let audio = Audio {
        sample_rate: 44100,
        planes: vec![signal(44100), vec![1.25; 44100]],
    };
    write_wav(&path, &audio).unwrap();
    let clip = decode(
        &path,
        DecodeOptions {
            start_seconds: 0.2,
            duration_seconds: Some(0.3),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(clip.samples(), 13230);
    assert_eq!(clip.planes[0], audio.planes[0][8820..22050]);
    assert!(clip.planes[1].iter().all(|&x| x == 1.25));
    assert!(
        decode(
            &path,
            DecodeOptions {
                start_seconds: 3.0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        decode(
            &path,
            DecodeOptions {
                max_samples_per_channel: 10,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn sinc_resampling_removes_delay_and_preserves_stereo_alignment() {
    let mut plane = vec![0.0; 48000];
    plane[24000] = 1.0;
    let audio = resample(
        Audio {
            sample_rate: 48000,
            planes: vec![plane.clone(), plane],
        },
        44100,
    )
    .unwrap();
    assert_eq!(audio.samples(), 44100);
    assert_eq!(audio.planes[0], audio.planes[1]);
    let peak = audio.planes[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap()
        .0;
    assert!(peak.abs_diff(22050) <= 1, "resampler delay: {peak}");
    let mono = Audio {
        sample_rate: 44100,
        planes: vec![signal(3000)],
    }
    .stereo()
    .unwrap();
    assert_eq!(mono.planes[0], mono.planes[1]);
    assert!(
        Audio {
            sample_rate: 44100,
            planes: vec![vec![0.]; 3]
        }
        .stereo()
        .is_err()
    );
}
