#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @unit_reduction(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<1x?x1x1xf32> into tensor<?xf32>
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"]} ins(%collapsed, %1 : tensor<?xf32>, tensor<f32>) outs(%2 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
    %expanded = tensor.expand_shape %3 [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
    return %expanded : tensor<1x1xf32>
  }
}

