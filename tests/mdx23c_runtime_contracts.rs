use ancha::{
    mdx23c_runtime::{Options, OverlapPlan, separate},
    spectral,
};
use ancha_audio::{
    Audio,
    decode::{DecodeOptions, decode},
    dsp::Stft,
};
use burn_flex::Flex;
use std::sync::atomic::AtomicBool;
#[path = "../crates/ancha-models/tests/support/mdx23c.rs"]
mod support;

#[test]
fn uvr_rectangular_padding_and_overlap_reconstruct_boundaries_and_exact_step_lengths() {
    assert!(OverlapPlan::new(usize::MAX, 128, 8).is_err());
    assert!(OverlapPlan::new(1, usize::MAX, 2).is_err());
    let native = OverlapPlan::new(132300, 261120, 8).unwrap();
    assert_eq!(
        (native.step, native.border, native.pad, native.chunks),
        (32640, 228480, 30900, 12)
    );
    for n in [1, 32, 127, 128, 129, 256, 333] {
        for overlap in [1, 2, 4, 8] {
            let p = OverlapPlan::new(n, 128, overlap).unwrap();
            let x = Audio {
                sample_rate: 44100,
                planes: vec![(0..n).map(|i| i as f32 * 0.001 + 0.1).collect()],
            };
            let mut y = vec![0f32; n];
            for i in 0..p.chunks {
                let start = i * p.step;
                let wave = p.chunk(&x, start);
                for pos in start.max(p.border)..(start + p.chunk).min(p.border + n) {
                    y[pos - p.border] += wave[0][pos - start];
                }
            }
            assert!(
                y.iter()
                    .zip(&x.planes[0])
                    .all(|(a, b)| (a / overlap as f32 - b).abs() < 1e-6)
            );
        }
    }
}

#[test]
fn mdx23c_retains_dc_and_stereo_complex_order_unlike_classic_mdx() {
    let x = vec![vec![0.25; 56], vec![-0.125; 56]];
    let mut stft = Stft::new(64, 8).unwrap();
    let (packed, frames) = spectral::pack(&mut stft, &x, 32, 0).unwrap();
    assert_eq!(frames, 8);
    assert!((packed[0] - 8.).abs() < 1e-6);
    assert!((packed[2 * 32 * frames] + 4.).abs() < 1e-6);
    let y = spectral::unpack(&mut stft, &packed, 32, frames, 56).unwrap();
    assert!(
        x.iter()
            .flatten()
            .zip(y.iter().flatten())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

#[test]
fn native_two_head_pipeline_preserves_length_and_never_publishes_cancelled_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    support::package(&model);
    let fixture = support::fixture();
    let manifest = ancha_models::mdx23c::read_manifest(&model).unwrap();
    let net =
        ancha_models::mdx23c::Mdx23c::<Flex>::load(&model, &manifest, &Default::default()).unwrap();
    let y = net
        .forward(
            burn::tensor::Tensor::from_data(
                burn::tensor::TensorData::new(support::input(), fixture.input_shape),
                &Default::default(),
            ),
            ancha_models::mdx23c::Options::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(y.dims(), fixture.output_shape);
    let data: Vec<f32> = y.into_data().to_vec().unwrap();
    support::assert_close(&data, &fixture.expected, 1e-8);
    assert!(
        data.iter()
            .zip(&fixture.expected)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
    let input = dir.path().join("source.wav");
    ancha_audio::write_wav(
        &input,
        &Audio {
            sample_rate: 44100,
            planes: vec![(0..113).map(|i| (i as f32 * 0.071).sin() * 0.3).collect()],
        },
    )
    .unwrap();
    let options = Options {
        input,
        model,
        output: dir.path().join("out"),
        decode: DecodeOptions::default(),
        chunk_samples: None,
        overlap: None,
        conv_gemm: false,
        optimized: true,
    };
    let r = separate::<Flex>(
        &options,
        &Default::default(),
        "cpu-flex",
        &AtomicBool::new(false),
        |_, _| {},
    )
    .unwrap();
    assert_eq!(r.samples_per_channel, 113);
    assert_eq!(r.profile, "native-context");
    assert_eq!(
        r.stems
            .iter()
            .map(|s| (s.name.as_str(), s.origin.as_str()))
            .collect::<Vec<_>>(),
        [("vocals", "predicted"), ("instrumental", "predicted")]
    );
    for name in ["vocals", "instrumental"] {
        let wave = decode(
            &options.output.join(format!("{name}.wav")),
            DecodeOptions::default(),
        )
        .unwrap();
        assert_eq!((wave.samples(), wave.planes.len()), (113, 2));
    }
    let mut bad = options.clone();
    bad.output = dir.path().join("cancel");
    assert!(
        separate::<Flex>(
            &bad,
            &Default::default(),
            "cpu-flex",
            &AtomicBool::new(true),
            |_, _| {}
        )
        .is_err()
    );
    assert!(!bad.output.exists());
    for chunk in [57, usize::MAX] {
        bad.chunk_samples = Some(chunk);
        assert!(
            separate::<Flex>(
                &bad,
                &Default::default(),
                "cpu-flex",
                &AtomicBool::new(false),
                |_, _| {}
            )
            .is_err()
        );
    }
    assert!(!bad.output.exists());
    bad.chunk_samples = None;
    bad.output = dir.path().join("during-cancel");
    let cancelled = AtomicBool::new(false);
    assert!(
        separate::<Flex>(
            &bad,
            &Default::default(),
            "cpu-flex",
            &cancelled,
            |done, _| if done == 1 {
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed)
            }
        )
        .is_err()
    );
    assert!(!bad.output.exists());
}
