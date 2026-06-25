module {
  func.func @copy_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.copy ins(%arg0 : tensor<8x16xf32>) outs(%arg1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @fill_2d(%arg0: f32, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.fill ins(%arg0 : f32) outs(%arg1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @exp_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.exp ins(%arg0 : tensor<8x16xf32>) outs(%arg1 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @add_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.add ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @sub_swapped(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.sub ins(%arg1, %arg0 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @div_2d(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.div ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}

