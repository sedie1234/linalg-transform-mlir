#map = affine_map<() -> ()>
module {
  func.func @detensorable_but_no_cf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
}

