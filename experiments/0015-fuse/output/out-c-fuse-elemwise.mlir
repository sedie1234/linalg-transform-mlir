module attributes {transform.with_named_sequence} {
  func.func @elem_chain(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c512 step %c64 iter_args(%arg3 = %arg1) -> (tensor<512x512xf32>) {
      %1 = scf.for %arg4 = %c0 to %c512 step %c64 iter_args(%arg5 = %arg3) -> (tensor<512x512xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
        %2 = linalg.elemwise_unary ins(%extracted_slice : tensor<64x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %extracted_slice_1 = tensor.extract_slice %arg0[%arg2, %arg4] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
        %extracted_slice_2 = tensor.extract_slice %arg5[%arg2, %arg4] [64, 64] [1, 1] : tensor<512x512xf32> to tensor<64x64xf32>
        %3 = linalg.elemwise_binary ins(%2, %extracted_slice_1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %inserted_slice = tensor.insert_slice %3 into %arg5[%arg2, %arg4] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<512x512xf32>
        scf.yield %inserted_slice : tensor<512x512xf32>
      }
      scf.yield %1 : tensor<512x512xf32>
    }
    return %0 : tensor<512x512xf32>
  }
}

