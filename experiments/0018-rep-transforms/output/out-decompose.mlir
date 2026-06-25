#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @sm(%arg0: tensor<2x16x32xf32>, %arg1: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
    %0 = tensor.empty() : tensor<2x16xf32>
    %cst = arith.constant -3.40282347E+38 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x16xf32>) -> tensor<2x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<2x16x32xf32>) outs(%1 : tensor<2x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.maxnumf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<2x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %2 : tensor<2x16x32xf32>, tensor<2x16xf32>) outs(%arg1 : tensor<2x16x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.subf %in, %in_1 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    } -> tensor<2x16x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<2x16xf32>) -> tensor<2x16xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3 : tensor<2x16x32xf32>) outs(%4 : tensor<2x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.addf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<2x16xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<2x16x32xf32>, tensor<2x16xf32>) outs(%arg1 : tensor<2x16x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.divf %in, %in_1 : f32
      linalg.yield %7 : f32
    } -> tensor<2x16x32xf32>
    return %6 : tensor<2x16x32xf32>
  }
}

