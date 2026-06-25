#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @two_ops(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %1 = scf.for %arg3 = %c0 to %c64 step %c16 iter_args(%arg4 = %0) -> (tensor<64x64xf32>) {
      %c0_0 = arith.constant 0 : index
      %c64_1 = arith.constant 64 : index
      %c16_2 = arith.constant 16 : index
      %2 = scf.for %arg5 = %c0_0 to %c64_1 step %c16_2 iter_args(%arg6 = %arg4) -> (tensor<64x64xf32>) {
        %c0_3 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c16_4 = arith.constant 16 : index
        %3 = scf.for %arg7 = %c0_3 to %c128 step %c16_4 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [16, 16] [1, 1] : tensor<64x128xf32> to tensor<16x16xf32>
          %extracted_slice_5 = tensor.extract_slice %arg1[%arg7, %arg5] [16, 16] [1, 1] : tensor<128x64xf32> to tensor<16x16xf32>
          %extracted_slice_6 = tensor.extract_slice %arg8[%arg3, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
          %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_5 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_6 : tensor<16x16xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %5 = arith.mulf %in, %in_7 : f32
            %6 = arith.addf %out, %5 : f32
            linalg.yield %6 : f32
          } -> tensor<16x16xf32>
          %inserted_slice = tensor.insert_slice %4 into %arg8[%arg3, %arg5] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
          scf.yield %inserted_slice : tensor<64x64xf32>
        }
        scf.yield %3 : tensor<64x64xf32>
      }
      scf.yield %2 : tensor<64x64xf32>
    }
    return %1 : tensor<64x64xf32>
  }
}

