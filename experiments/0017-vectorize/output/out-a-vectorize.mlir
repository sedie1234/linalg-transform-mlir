#map = affine_map<(d0, d1) -> (d0, 0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1, d0)>
module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<8x16xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : tensor<8x16xf32>, vector<8x4x16xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : tensor<16x4xf32>, vector<8x4x16xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x4xf32>, vector<8x4xf32>
    %3 = arith.mulf %0, %1 : vector<8x4x16xf32>
    %4 = vector.multi_reduction <add>, %3, %2 [2] : vector<8x4x16xf32> to vector<8x4xf32>
    %5 = vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, tensor<8x4xf32>
    return %5 : tensor<8x4xf32>
  }
}

