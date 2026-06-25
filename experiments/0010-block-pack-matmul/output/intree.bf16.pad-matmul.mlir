#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @pad_matmul(%arg0: tensor<30x30xf32>, %arg1: tensor<30x30xf32>, %arg2: tensor<30x30xf32>) -> tensor<30x30xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x2x16x16xf32>
    %pack = tensor.pack %arg0 padding_value(%cst : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<30x30xf32> -> tensor<2x2x16x16xf32>
    %1 = tensor.empty() : tensor<2x2x16x16xf32>
    %pack_0 = tensor.pack %arg1 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 16] into %1 : tensor<30x30xf32> -> tensor<2x2x16x16xf32>
    %2 = tensor.empty() : tensor<2x2x16x16xf32>
    %pack_1 = tensor.pack %arg2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<30x30xf32> -> tensor<2x2x16x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<2x2x16x16xf32>, tensor<2x2x16x16xf32>) outs(%pack_1 : tensor<2x2x16x16xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<2x2x16x16xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg2 : tensor<2x2x16x16xf32> -> tensor<30x30xf32>
    return %unpack : tensor<30x30xf32>
  }
}

