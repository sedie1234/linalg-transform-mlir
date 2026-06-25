module {
  func.func @matmul_f32(%arg0: tensor<64x256xf32>, %arg1: tensor<256x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}

