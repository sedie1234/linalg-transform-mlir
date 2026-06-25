#map = affine_map<(d0) -> (d0)>
module {
  func.func @inline_rank0(%arg0: tensor<f32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %extracted = tensor.extract %arg0[] : tensor<f32>
      %2 = arith.addf %extracted, %in : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}

