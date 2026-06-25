module attributes {transform.with_named_sequence} {
  func.func @mm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
