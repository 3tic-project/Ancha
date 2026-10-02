use ancha::runtime::residual_audio;
use ancha_audio::Audio;
mod support;

#[test]
fn synthetic_model_runs_full_pipeline_both_families_and_preserves_native_stems() {
    use ancha::runtime::{SeparateOptions, separate};
    use ancha_audio::decode::{DecodeOptions, decode};
    use ancha_models::{config::Family, roformer::AttentionPlan};
    use std::sync::atomic::AtomicBool;
    for family in [Family::BsRoformer, Family::MelBandRoformer] {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        support::tiny_package(&model, family);
        let input = temp.path().join("mono.wav");
        let signal = (0..5003).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();
        ancha_audio::write_wav(
            &input,
            &Audio {
                sample_rate: 44100,
                planes: vec![signal],
            },
        )
        .unwrap();
        let output = temp.path().join("output");
        let options = SeparateOptions {
            input,
            model,
            output: output.clone(),
            decode: DecodeOptions::default(),
            chunk_samples: None,
            overlap: None,
            attention: AttentionPlan {
                query_tile: 3,
                group_tile: 2,
                batched_linear: false,
            },
            max_score_mib: 1,
        };
        let mut progress = Vec::new();
        let report = separate::<burn::backend::NdArray<f32>>(
            &options,
            &Default::default(),
            "cpu-ndarray",
            &AtomicBool::new(false),
            |n, total| progress.push((n, total)),
        )
        .unwrap();
        assert_eq!(report.samples_per_channel, 5003);
        assert_eq!(report.channels, 2);
        assert_eq!(report.chunks, 5); // reflected borders and non-aligned tail
        assert_eq!(progress.last().unwrap(), &(5, 5));
        assert_eq!(
            report.stems[1].origin,
            if family == Family::BsRoformer {
                "residual"
            } else {
                "predicted"
            }
        );
        assert_eq!(
            report.residual_reconstruction_max_abs.is_some(),
            family == Family::BsRoformer
        );
        for name in ["vocals", "instrumental"] {
            let audio = decode(
                &output.join(format!("{name}.wav")),
                DecodeOptions::default(),
            )
            .unwrap();
            assert_eq!(audio.samples(), 5003);
            assert_eq!(audio.planes.len(), 2);
            audio.validate().unwrap();
        }
        assert!(output.join("run.json").is_file());
        let baseline_options = SeparateOptions {
            output: temp.path().join("batched"),
            attention: AttentionPlan {
                batched_linear: true,
                ..options.attention
            },
            ..options.clone()
        };
        separate::<burn::backend::NdArray<f32>>(
            &baseline_options,
            &Default::default(),
            "cpu",
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        for name in ["vocals", "instrumental"] {
            let flat = decode(
                &output.join(format!("{name}.wav")),
                DecodeOptions::default(),
            )
            .unwrap();
            let batched = decode(
                &baseline_options.output.join(format!("{name}.wav")),
                DecodeOptions::default(),
            )
            .unwrap();
            let max = flat
                .planes
                .iter()
                .flatten()
                .zip(batched.planes.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max < 1e-6, "projection layout changed {name}: {max}");
        }
        assert!(
            separate::<burn::backend::NdArray<f32>>(
                &options,
                &Default::default(),
                "cpu",
                &AtomicBool::new(false),
                |_, _| {}
            )
            .is_err()
        );
        let options = SeparateOptions {
            output: temp.path().join("cancelled"),
            ..options
        };
        assert!(
            separate::<burn::backend::NdArray<f32>>(
                &options,
                &Default::default(),
                "cpu",
                &AtomicBool::new(true),
                |_, _| {}
            )
            .is_err()
        );
        assert!(!options.output.exists());
    }
}

#[test]
fn residual_keeps_channel_order_and_reconstructs_unclipped_mixture() {
    let source = Audio {
        sample_rate: 44100,
        planes: vec![vec![0.3, 1.8, -1.2], vec![-0.2, 0.4, 0.5]],
    };
    let prediction = Audio {
        sample_rate: 44100,
        planes: vec![vec![0.1, 0.2, 0.4], vec![0.2, 0.3, -0.1]],
    };
    let residual = residual_audio(&source, &prediction).unwrap();
    for ((x, y), r) in source
        .planes
        .iter()
        .flatten()
        .zip(prediction.planes.iter().flatten())
        .zip(residual.planes.iter().flatten())
    {
        assert!((x - (y + r)).abs() < 2e-7);
    }
    let bad = Audio {
        sample_rate: 48000,
        ..prediction
    };
    assert!(residual_audio(&source, &bad).is_err());
}
