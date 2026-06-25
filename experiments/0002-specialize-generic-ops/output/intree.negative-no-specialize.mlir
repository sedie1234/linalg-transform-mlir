#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @transpose_like(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<16x8xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x16xf32>) outs(%arg1 : tensor<16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
  func.func @fused_exp_neg(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x16xf32>) outs(%arg1 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.negf %in : f32
      %2 = math.exp %1 : f32
      linalg.yield %2 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @max_elemwise(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%arg2 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.maximumf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @multi_k_contract(%arg0: tensor<10x20x30xf32>, %arg1: tensor<30x20x40xf32>, %arg2: tensor<10x40xf32>) -> tensor<10x40xf32> {
    %0 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<30x20x40xf32>) outs(%arg2 : tensor<10x40xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<10x40xf32>
    return %0 : tensor<10x40xf32>
  }
}

