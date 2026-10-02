#![cfg(feature = "onnx")]
use ancha::mdx_runtime::{MdxOptions, pack, separate_mdx, unpack};
use ancha_audio::{decode::DecodeOptions, dsp::Stft};
use burn::backend::NdArray;
use std::sync::atomic::AtomicBool;
#[test]
fn mdx_channel_complex_packing_zero_low_bins_and_frequency_padding_match_stft() {
    for fft in [5120, 6144] {
        let n = 10240;
        let planes: Vec<Vec<f32>> = (0..2)
            .map(|c| {
                (0..n)
                    .map(|i| (i as f32 * (0.015 + c as f32 * 0.09)).sin() * 0.1)
                    .collect()
            })
            .collect();
        let bins = 2048;
        let mut stft = Stft::new(fft, 1024).unwrap();
        let (packed, frames) = pack(&mut stft, &planes, bins).unwrap();
        assert_eq!(packed.len(), 4 * bins * frames);
        let actual = unpack(&mut stft, &packed, bins, frames, n).unwrap();
        for c in 0..2 {
            let mut z = stft.forward(&planes[c]).unwrap();
            for t in 0..frames {
                for f in 0..z.bins {
                    if f < 3 || f >= bins {
                        z.data[t * z.bins + f] = Default::default();
                    }
                }
            }
            let expected = stft.inverse(&z, n, false).unwrap();
            assert!(
                actual[c]
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 1e-7)
            );
        }
    }
}
#[test]
fn mdx_invalid_overlap_and_cancellation_publish_no_outputs() {
    let t = tempfile::tempdir().unwrap();
    let o = MdxOptions {
        input: t.path().join("unused.wav"),
        model: t.path().join("unused.onnx"),
        output: t.path().join("out"),
        decode: DecodeOptions::default(),
        overlap: Some(0.99),
        denoise: false,
        optimized: true,
        batch_size: 1,
        conv_gemm: false,
    };
    assert!(
        separate_mdx::<NdArray<f32>>(
            &o,
            &Default::default(),
            "cpu",
            &AtomicBool::new(false),
            |_, _| {}
        )
        .is_err()
    );
    assert!(!o.output.exists());
    let mut o = o;
    o.overlap = None;
    assert!(
        separate_mdx::<NdArray<f32>>(
            &o,
            &Default::default(),
            "cpu",
            &AtomicBool::new(true),
            |_, _| {}
        )
        .is_err()
    );
    assert!(!o.output.exists());
}
