#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_mul_sub_chain(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %in_1 : f32
      %4 = arith.subf %3, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}

