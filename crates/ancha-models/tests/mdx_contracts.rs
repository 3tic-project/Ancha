#![cfg(feature = "onnx")]
use ancha_models::mdx::Graph as Executable;
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use onnx_rs::ast::*;
use std::sync::atomic::AtomicBool;
type B = NdArray<f32>;
fn value(name: &str) -> ValueInfo<'_> {
    ValueInfo {
        name,
        ..Default::default()
    }
}
fn node<'a>(op: OpType<'a>, input: Vec<&'a str>, output: &'a str) -> Node<'a> {
    Node {
        op_type: op,
        input,
        output: vec![output],
        ..Default::default()
    }
}
fn model(graph: Graph<'_>) -> Vec<u8> {
    onnx_rs::encode(&Model {
        ir_version: 7,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 13,
        }],
        graph: Some(graph),
        ..Default::default()
    })
}
#[test]
fn transpose_bn_folding_matches_eval_reference() {
    let bytes = model(Graph {
        input: vec![value("x")],
        output: vec![value("out")],
        initializer: vec![
            TensorProto::from_f32("w", vec![1, 2, 1, 1], vec![2., -1.]),
            TensorProto::from_f32("bias", vec![2], vec![0.25, -0.75]),
            TensorProto::from_f32("gamma", vec![2], vec![1.5, 0.5]),
            TensorProto::from_f32("beta", vec![2], vec![0.4, -0.3]),
            TensorProto::from_f32("mean", vec![2], vec![0.8, -1.2]),
            TensorProto::from_f32("var", vec![2], vec![0.5, 2.]),
        ],
        node: vec![
            node(OpType::ConvTranspose, vec!["x", "w", "bias"], "conv"),
            node(
                OpType::BatchNormalization,
                vec!["conv", "gamma", "beta", "mean", "var"],
                "bn",
            ),
            node(OpType::Relu, vec!["bn"], "out"),
        ],
        ..Default::default()
    });
    let d = Default::default();
    let input =
        || Tensor::<B, 4>::from_data(TensorData::new(vec![1., -2., 0., 4.], [1, 1, 2, 2]), &d);
    let reference = Executable::<B>::from_bytes(&bytes, false, &d).unwrap();
    let optimized = Executable::<B>::from_bytes(&bytes, true, &d).unwrap();
    assert_eq!(reference.folded_bn, 0);
    assert_eq!(optimized.folded_bn, 1);
    let a: Vec<f32> = reference
        .forward(input(), &AtomicBool::new(false))
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap();
    let b: Vec<f32> = optimized
        .forward(input(), &AtomicBool::new(false))
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap();
    for c in 0..2 {
        for (i, &x) in [1., -2., 0., 4.].iter().enumerate() {
            let expected = (((x * [2., -1.][c] + [0.25, -0.75][c]) - [0.8, -1.2][c])
                / ([0.5, 2.][c] + 1e-5f32).sqrt()
                * [1.5, 0.5][c]
                + [0.4, -0.3][c])
                .max(0.);
            assert!((a[c * 4 + i] - expected).abs() < 2e-6);
            assert!((b[c * 4 + i] - expected).abs() < 2e-6);
        }
    }
}
#[test]
fn nchw_matmul_broadcast_transpose_conv_and_live_skip_values_match_scalars() {
    let mut transpose = node(OpType::Transpose, vec!["sum"], "transposed");
    transpose.attribute.push(Attribute {
        name: "perm",
        ints: vec![0, 1, 3, 2],
        r#type: AttributeType::Ints,
        ..Default::default()
    });
    let bytes = model(Graph {
        input: vec![value("x")],
        output: vec![value("out")],
        initializer: vec![
            TensorProto::from_f32("matrix", vec![2, 2], vec![1., 2., 3., 4.]),
            TensorProto::from_f32("scale", vec![1], vec![0.5]),
            TensorProto::from_f32("w", vec![1, 1, 1, 1], vec![2.]),
        ],
        node: vec![
            node(OpType::MatMul, vec!["x", "matrix"], "product"),
            node(OpType::Add, vec!["product", "x"], "sum"),
            transpose,
            node(OpType::Mul, vec!["transposed", "scale"], "scaled"),
            node(OpType::Conv, vec!["scaled", "w"], "out"),
        ],
        ..Default::default()
    });
    let d = Default::default();
    let graph = Executable::<B>::from_bytes(&bytes, true, &d).unwrap();
    let input = Tensor::from_data(TensorData::new(vec![1., 2., 3., 4.], [1, 1, 2, 2]), &d);
    let actual: Vec<f32> = graph
        .forward(input, &AtomicBool::new(false))
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap();
    assert_eq!(actual, vec![8., 18., 12., 26.]);
    let batched = Tensor::from_data(
        TensorData::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], [2, 1, 2, 2]),
        &d,
    );
    let actual: Vec<f32> = graph
        .forward(batched, &AtomicBool::new(false))
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap();
    assert_eq!(actual, vec![8., 18., 12., 26., 28., 38., 40., 54.]);
}
#[test]
fn graph_rejects_unknown_ops_nonfinite_constants_and_cancels() {
    let g = Graph {
        input: vec![value("x")],
        output: vec![value("out")],
        node: vec![node(OpType::Relu, vec!["x"], "out")],
        ..Default::default()
    };
    let d = Default::default();
    let graph = Executable::<B>::from_bytes(&model(g.clone()), true, &d).unwrap();
    assert!(
        graph
            .forward(Tensor::zeros([1, 1, 1, 1], &d), &AtomicBool::new(true))
            .is_err()
    );
    let mut bad = g.clone();
    bad.node[0].op_type = OpType::Sigmoid;
    assert!(Executable::<B>::from_bytes(&model(bad), true, &d).is_err());
    let mut bad = g;
    bad.initializer
        .push(TensorProto::from_f32("invalid", vec![1], vec![f32::NAN]));
    assert!(Executable::<B>::from_bytes(&model(bad), true, &d).is_err());
}
