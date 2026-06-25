// sched-D: decompose — linalg.softmax → fill/max/exp/sum/div generic 시퀀스.
// softmax 는 AggregatedOpInterface 를 통해 decompose_interface 로 풀린다.
module attributes {transform.with_named_sequence} {
  func.func @sm(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
    %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
    return %1 : tensor<2x16x32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %sm = transform.structured.match ops{["linalg.softmax"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %d = transform.structured.decompose_interface %sm : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
