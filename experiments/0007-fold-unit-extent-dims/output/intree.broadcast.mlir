#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func @broadcast_add(%arg0: tensor<1x5xf32>, %arg1: tensor<5x1xf32>) -> tensor<5x5xf32> {
    %0 = tensor.empty() : tensor<5x5xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x5xf32> into tensor<5xf32>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<5x1xf32> into tensor<5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %collapsed_0 : tensor<5xf32>, tensor<5xf32>) outs(%0 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<5x5xf32>
    return %1 : tensor<5x5xf32>
  }
  func.func @drop_unit_loop_with_index(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x8xf32> into tensor<8xf32>
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%collapsed : tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.addf %in, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<8xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
    return %expanded : tensor<1x8xf32>
  }
}

