module {
  func.func @matmul_f32(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
  }
  func.func @matmul_transpose_b(%arg0: tensor<16x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
  }
  func.func @matmul_transpose_a(%arg0: tensor<8x16xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
  }
  func.func @batch_matmul_i32(%arg0: tensor<4x16x8xi32>, %arg1: tensor<4x8x32xi32>, %arg2: tensor<4x16x32xi32>) -> tensor<4x16x32xi32> {
    %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x16x8xi32>, tensor<4x8x32xi32>) outs(%arg2 : tensor<4x16x32xi32>) -> tensor<4x16x32xi32>
    return %0 : tensor<4x16x32xi32>
  }
}

