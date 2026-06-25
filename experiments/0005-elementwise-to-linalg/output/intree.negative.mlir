module {
  func.func @scalar_add(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @vector_add(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
    %0 = arith.addf %arg0, %arg1 : vector<4xf32>
    return %0 : vector<4xf32>
  }
  func.func @select_scalar_cond(%arg0: i1, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = arith.select %arg0, %arg1, %arg2 : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

