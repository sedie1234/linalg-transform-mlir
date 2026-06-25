#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @no_inline_memref(%arg0: memref<f32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) {
    linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<f32>, memref<8xf32>) outs(%arg2 : memref<8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return
  }
  func.func @no_inline_nonconst(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}

