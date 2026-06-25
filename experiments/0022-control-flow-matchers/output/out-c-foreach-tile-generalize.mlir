#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @multi(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64x96xf32>, %arg3: tensor<96x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %c96 = arith.constant 96 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %0) -> (tensor<64x64xf32>) {
      %3 = scf.for %arg7 = %c0 to %c64 step %c32 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>) {
        %4 = scf.for %arg9 = %c0 to %c128 step %c32 iter_args(%arg10 = %arg8) -> (tensor<64x64xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg5, %arg9] [32, 32] [1, 1] : tensor<64x128xf32> to tensor<32x32xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg9, %arg7] [32, 32] [1, 1] : tensor<128x64xf32> to tensor<32x32xf32>
          %extracted_slice_1 = tensor.extract_slice %arg10[%arg5, %arg7] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
          %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_2: f32, %out: f32):
            %6 = arith.mulf %in, %in_2 : f32
            %7 = arith.addf %out, %6 : f32
            linalg.yield %7 : f32
          } -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %5 into %arg10[%arg5, %arg7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
          scf.yield %inserted_slice : tensor<64x64xf32>
        }
        scf.yield %4 : tensor<64x64xf32>
      }
      scf.yield %3 : tensor<64x64xf32>
    }
    %2 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %0) -> (tensor<64x64xf32>) {
      %3 = scf.for %arg7 = %c0 to %c64 step %c32 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>) {
        %4 = scf.for %arg9 = %c0 to %c96 step %c32 iter_args(%arg10 = %arg8) -> (tensor<64x64xf32>) {
          %extracted_slice = tensor.extract_slice %arg2[%arg5, %arg9] [32, 32] [1, 1] : tensor<64x96xf32> to tensor<32x32xf32>
          %extracted_slice_0 = tensor.extract_slice %arg3[%arg9, %arg7] [32, 32] [1, 1] : tensor<96x64xf32> to tensor<32x32xf32>
          %extracted_slice_1 = tensor.extract_slice %arg10[%arg5, %arg7] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
          %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_2: f32, %out: f32):
            %6 = arith.mulf %in, %in_2 : f32
            %7 = arith.addf %out, %6 : f32
            linalg.yield %7 : f32
          } -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %5 into %arg10[%arg5, %arg7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
          scf.yield %inserted_slice : tensor<64x64xf32>
        }
        scf.yield %4 : tensor<64x64xf32>
      }
      scf.yield %3 : tensor<64x64xf32>
    }
    return %1, %2 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}

