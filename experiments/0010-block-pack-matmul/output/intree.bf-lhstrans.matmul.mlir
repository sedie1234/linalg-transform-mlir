#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @matmul_f32(%arg0: tensor<64x256xf32>, %arg1: tensor<256x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<4x2x64x32xf32>
    %pack = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [64, 32] into %0 : tensor<64x256xf32> -> tensor<4x2x64x32xf32>
    %1 = tensor.empty() : tensor<8x4x16x64xf32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64] into %1 : tensor<256x128xf32> -> tensor<8x4x16x64xf32>
    %2 = tensor.empty() : tensor<2x8x32x16xf32>
    %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %2 : tensor<64x128xf32> -> tensor<2x8x32x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x2x64x32xf32>, tensor<8x4x16x64xf32>) outs(%pack_1 : tensor<2x8x32x16xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<2x8x32x16xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %arg2 : tensor<2x8x32x16xf32> -> tensor<64x128xf32>
    return %unpack : tensor<64x128xf32>
  }
}

