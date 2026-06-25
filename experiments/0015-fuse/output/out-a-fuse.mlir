module attributes {transform.with_named_sequence} {
  func.func @mm_chain(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256x64xf32>, %arg3: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<128x64xf32>) {
      %1 = scf.for %arg6 = %c0 to %c64 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x64xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %extracted_slice_1 = tensor.extract_slice %arg0[%arg4, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
        %2 = linalg.elemwise_binary ins(%extracted_slice, %extracted_slice_0 : tensor<32x256xf32>, tensor<32x256xf32>) outs(%extracted_slice_1 : tensor<32x256xf32>) -> tensor<32x256xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[0, %arg6] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
        %extracted_slice_3 = tensor.extract_slice %arg7[%arg4, %arg6] [32, 32] [1, 1] : tensor<128x64xf32> to tensor<32x32xf32>
        %3 = linalg.matmul ins(%2, %extracted_slice_2 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%extracted_slice_3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %inserted_slice = tensor.insert_slice %3 into %arg7[%arg4, %arg6] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
        scf.yield %inserted_slice : tensor<128x64xf32>
      }
      scf.yield %1 : tensor<128x64xf32>
    }
    return %0 : tensor<128x64xf32>
  }
}

