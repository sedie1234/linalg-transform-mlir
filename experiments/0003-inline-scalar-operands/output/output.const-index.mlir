#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inline_const_index(%arg0: tensor<2x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<4x5xf32>) outs(%0 : tensor<4x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %extracted = tensor.extract %arg0[%c0, %c1] : tensor<2x3xf32>
      %2 = arith.mulf %extracted, %in : f32
      linalg.yield %2 : f32
    } -> tensor<4x5xf32>
    return %1 : tensor<4x5xf32>
  }
}

