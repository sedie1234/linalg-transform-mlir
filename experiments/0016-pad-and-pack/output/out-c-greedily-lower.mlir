#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
module {
  func.func @matmul_mk_kn_mn(%arg0: tensor<1023x255xf32>, %arg1: tensor<255x127xf32>, %arg2: tensor<1023x127xf32>) -> tensor<1023x127xf32> {
    %0 = tensor.empty() : tensor<128x8x32x8xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0] high[1, 1] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1023x255xf32> to tensor<1024x256xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] output_shape [128, 8, 8, 32] : tensor<1024x256xf32> into tensor<128x8x8x32xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<128x8x8x32xf32>) outs(%0 : tensor<128x8x32x8xf32>) permutation = [0, 2, 3, 1] 
    %1 = tensor.empty() : tensor<8x8x32x16xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %pack = tensor.pack %arg1 padding_value(%cst_0 : f32) inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %1 : tensor<255x127xf32> -> tensor<8x8x32x16xf32>
    %2 = tensor.empty() : tensor<128x8x8x16xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %pack_2 = tensor.pack %arg2 padding_value(%cst_1 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %2 : tensor<1023x127xf32> -> tensor<128x8x8x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%transposed, %pack : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>) outs(%pack_2 : tensor<128x8x8x16xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %4 = arith.mulf %in, %in_3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<128x8x8x16xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %arg2 : tensor<128x8x8x16xf32> -> tensor<1023x127xf32>
    return %unpack : tensor<1023x127xf32>
  }
}

