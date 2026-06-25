// sched-A: generalize — linalg.matmul → linalg.generic (named → generic).
module attributes {transform.with_named_sequence} {
  func.func @mm(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
                       outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %g = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
