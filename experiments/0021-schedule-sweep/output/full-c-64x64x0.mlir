module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %0 = scf.for %arg3 = %c0 to %c128 step %c64 iter_args(%arg4 = %arg2) -> (tensor<128x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [64, 256] [1, 1] : tensor<128x256xf32> to tensor<64x256xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[%arg3, 0] [64, 64] [1, 1] : tensor<128x64xf32> to tensor<64x64xf32>
      %1 = linalg.matmul ins(%extracted_slice, %arg1 : tensor<64x256xf32>, tensor<256x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %inserted_slice = tensor.insert_slice %1 into %arg4[%arg3, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x64xf32>
      scf.yield %inserted_slice : tensor<128x64xf32>
    }
    return %0 : tensor<128x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [64, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
  }
}

