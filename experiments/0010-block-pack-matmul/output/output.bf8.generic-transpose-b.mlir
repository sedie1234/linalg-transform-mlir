#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @generic_transpose_b(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = tensor.empty() : tensor<4x4x8x8xf32>
    %pack = tensor.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0 : tensor<32x32xf32> -> tensor<4x4x8x8xf32>
    %1 = tensor.empty() : tensor<4x4x8x8xf32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %1 : tensor<32x32xf32> -> tensor<4x4x8x8xf32>
    %2 = tensor.empty() : tensor<4x4x8x8xf32>
    %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %2 : tensor<32x32xf32> -> tensor<4x4x8x8xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x4x8x8xf32>, tensor<4x4x8x8xf32>) outs(%pack_1 : tensor<4x4x8x8xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<4x4x8x8xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %arg2 : tensor<4x4x8x8xf32> -> tensor<32x32xf32>
    return %unpack : tensor<32x32xf32>
  }
}

