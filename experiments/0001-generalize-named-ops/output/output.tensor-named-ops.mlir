#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @tensor_named(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x16xf32>, %arg3: tensor<4x16xf32>, %arg4: tensor<4x16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<16x4xf32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<4x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<4x16xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg4 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%arg2 : tensor<4x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4x16xf32>
    %2 = tensor.empty() : tensor<16x4xf32>
    %3 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x16xf32>) outs(%2 : tensor<16x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x4xf32>
    return %1, %0, %3 : tensor<4x16xf32>, tensor<4x16xf32>, tensor<16x4xf32>
  }
}

