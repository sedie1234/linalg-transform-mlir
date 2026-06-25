module attributes {transform.with_named_sequence} {
  func.func @two_ops(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %g = transform.structured.generalize %t : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
