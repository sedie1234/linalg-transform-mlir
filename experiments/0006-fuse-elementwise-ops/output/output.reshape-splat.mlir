#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @collapse_into_generic(%arg0: tensor<2x3x4xf32>, %arg1: tensor<6x4xf32>) -> tensor<6x4xf32> {
    %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] output_shape [2, 3, 4] : tensor<6x4xf32> into tensor<2x3x4xf32>
    %0 = tensor.empty() : tensor<2x3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %expanded : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%0 : tensor<2x3x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x3x4xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
    return %collapsed : tensor<6x4xf32>
  }
  func.func @splat_fold(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.mulf %in, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}

