module attributes {transform.with_named_sequence} {
  func.func @two_gen(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<32xf32>
    %2 = tensor.empty() : tensor<32xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<32xf32>
    return %3 : tensor<32xf32>
  }
}
