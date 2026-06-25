#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @static_elementwise(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xi1>, tensor<8x16xf32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%arg0 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<8x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %arg0 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%0 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.mulf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<8x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<8x16xf32>) outs(%1 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.exp %in : f32
      linalg.yield %6 : f32
    } -> tensor<8x16xf32>
    %3 = tensor.empty() : tensor<8x16xi1>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%3 : tensor<8x16xi1>) {
    ^bb0(%in: f32, %in_0: f32, %out: i1):
      %6 = arith.cmpf ogt, %in, %in_0 : f32
      linalg.yield %6 : i1
    } -> tensor<8x16xi1>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %2, %arg1 : tensor<8x16xi1>, tensor<8x16xf32>, tensor<8x16xf32>) outs(%2 : tensor<8x16xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %6 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<8x16xf32>
    return %2, %4, %5 : tensor<8x16xf32>, tensor<8x16xi1>, tensor<8x16xf32>
  }
}

