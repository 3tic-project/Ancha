use ancha_models::{
    config::ModelConfig,
    spatial::{InstanceNorm, frequency_shuffle, resize},
};
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
type B = NdArray<f32>;
#[test]
fn instance_norm_uses_spatial_biased_variance_and_input_statistics() {
    let d = Default::default();
    let norm = InstanceNorm::<B>::from_affine(
        Tensor::from_floats([2., 3.], &d),
        Tensor::from_floats([0.5, -1.], &d),
    );
    let x = vec![1., 2., 3., 4., 5., 6., 10., 12., 14., 16., 18., 20.];
    let actual: Vec<f32> = norm
        .forward(Tensor::from_data(
            TensorData::new(x.clone(), [1, 2, 2, 3]),
            &d,
        ))
        .into_data()
        .to_vec()
        .unwrap();
    for c in 0..2 {
        let v = &x[c * 6..c * 6 + 6];
        let mean = v.iter().sum::<f32>() / 6.;
        let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 6.;
        for i in 0..6 {
            let expected = (v[i] - mean) / (variance + 1e-8).sqrt() * [2., 3.][c] + [0.5, -1.][c];
            assert!((actual[c * 6 + i] - expected).abs() < 2e-6);
        }
    }
    let constant: Vec<f32> = norm
        .forward(Tensor::ones([1, 2, 2, 3], &d))
        .into_data()
        .to_vec()
        .unwrap();
    assert_eq!(&constant[..6], &[0.5; 6]);
    assert_eq!(&constant[6..], &[-1.; 6]);
}
#[test]
fn bilinear_half_pixel_edges_and_odd_sizes_match_scalar_reference() {
    let d = Default::default();
    let data: Vec<f32> = (0..15).map(|v| (v * v) as f32).collect();
    for [oh, ow] in [[7, 9], [2, 3], [1, 1], [3, 5]] {
        let actual: Vec<f32> = resize(
            Tensor::<B, 4>::from_data(TensorData::new(data.clone(), [1, 1, 3, 5]), &d),
            [oh, ow],
        )
        .into_data()
        .to_vec()
        .unwrap();
        for y in 0..oh {
            for x in 0..ow {
                let fy = ((y as f64 + 0.5) * 3. / oh as f64 - 0.5).max(0.);
                let fx = ((x as f64 + 0.5) * 5. / ow as f64 - 0.5).max(0.);
                let y0 = (fy.floor() as usize).min(2);
                let x0 = (fx.floor() as usize).min(4);
                let wy = (fy - y0 as f64) as f32;
                let wx = (fx - x0 as f64) as f32;
                let a = data[y0 * 5 + x0];
                let b = data[y0 * 5 + (x0 + 1).min(4)];
                let c = data[(y0 + 1).min(2) * 5 + x0];
                let e = data[(y0 + 1).min(2) * 5 + (x0 + 1).min(4)];
                let expected = (a * (1. - wx) + b * wx) * (1. - wy) + (c * (1. - wx) + e * wx) * wy;
                assert!((actual[y * ow + x] - expected).abs() < 3e-5);
            }
        }
    }
}
#[test]
fn shuffle_expands_only_frequency_and_preserves_phase_order() {
    let d = Default::default();
    let y = frequency_shuffle(
        Tensor::<B, 4>::from_data(
            TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), [1, 4, 2, 3]),
            &d,
        ),
        2,
    );
    assert_eq!(y.dims(), [1, 2, 2, 6]);
    let v: Vec<f32> = y.into_data().to_vec().unwrap();
    assert_eq!(
        &v[..12],
        &[0., 6., 1., 7., 2., 8., 3., 9., 4., 10., 5., 11.]
    );
    assert_eq!(
        &v[12..],
        &[12., 18., 13., 19., 14., 20., 15., 21., 16., 22., 17., 23.]
    );
}
#[test]
fn hyperace_preset_has_full_band_partition_and_native_contract() {
    for instrumental in [false, true] {
        let c = ModelConfig::hyperace_v2(instrumental);
        c.validate().unwrap();
        assert_eq!(c.bands.len(), 62);
        assert_eq!(
            c.bands.iter().flatten().copied().collect::<Vec<_>>(),
            (0..1025).collect::<Vec<_>>()
        );
        assert!(!c.zero_dc);
        assert_eq!(c.chunk_samples, 960000);
        assert_eq!(c.overlap, 4);
        let mut invalid = c.clone();
        invalid.dim = 128;
        assert!(invalid.validate().is_err());
    }
}
