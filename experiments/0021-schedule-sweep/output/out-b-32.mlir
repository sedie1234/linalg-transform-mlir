module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %0 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128x64xf32>) {
      %1 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %arg4) -> (tensor<128x64xf32>) {
        %2 = scf.for %arg7 = %c0 to %c256 step %c32 iter_args(%arg8 = %arg6) -> (tensor<128x64xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [32, 32] [1, 1] : tensor<128x256xf32> to tensor<32x32xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg7, %arg5] [32, 32] [1, 1] : tensor<256x64xf32> to tensor<32x32xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<128x64xf32> to tensor<32x32xf32>
          %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %3 into %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
          scf.yield %inserted_slice : tensor<128x64xf32>
        }
        scf.yield %2 : tensor<128x64xf32>
      }
      scf.yield %1 : tensor<128x64xf32>
    }
    return %0 : tensor<128x64xf32>
  }
}

