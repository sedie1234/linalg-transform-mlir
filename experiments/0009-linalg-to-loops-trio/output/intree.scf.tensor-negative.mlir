module {
  func.func @matmul_tensor(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

