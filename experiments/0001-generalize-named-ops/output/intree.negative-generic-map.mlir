#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @negative(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<4x16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%arg2 : tensor<4x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<4x16xf32>
    %mapped = linalg.map { arith.addf } ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%arg2 : tensor<4x16xf32>)
    return %0, %mapped : tensor<4x16xf32>, tensor<4x16xf32>
  }
}

