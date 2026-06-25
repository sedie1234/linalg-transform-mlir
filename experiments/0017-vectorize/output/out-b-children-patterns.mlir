#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<8x16xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x16xf32>, vector<8x16xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x4xf32>, vector<16x4xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x4xf32>, vector<8x4xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<8x16xf32>, vector<16x4xf32> into vector<8x4xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, tensor<8x4xf32>
    return %4 : tensor<8x4xf32>
  }
}

