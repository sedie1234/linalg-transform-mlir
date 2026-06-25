#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module attributes {transform.with_named_sequence} {
  func.func @mm_chain(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256x64xf32>, %arg3: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = scf.forall (%arg4, %arg5) in (4, 4) shared_outs(%arg6 = %arg3) -> (tensor<128x64xf32>) {
      %1 = affine.apply #map(%arg5)
      %2 = affine.apply #map1(%arg4)
      %3 = affine.apply #map(%arg5)
      %4 = affine.apply #map1(%arg4)
      %5 = affine.apply #map1(%arg4)
      %6 = affine.apply #map1(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%4, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%5, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
      %extracted_slice_1 = tensor.extract_slice %arg0[%6, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
      %7 = linalg.elemwise_binary ins(%extracted_slice, %extracted_slice_0 : tensor<32x256xf32>, tensor<32x256xf32>) outs(%extracted_slice_1 : tensor<32x256xf32>) -> tensor<32x256xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[0, %1] [256, 16] [1, 1] : tensor<256x64xf32> to tensor<256x16xf32>
      %extracted_slice_3 = tensor.extract_slice %arg6[%2, %3] [32, 16] [1, 1] : tensor<128x64xf32> to tensor<32x16xf32>
      %8 = linalg.matmul ins(%7, %extracted_slice_2 : tensor<32x256xf32>, tensor<256x16xf32>) outs(%extracted_slice_3 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %9 = affine.apply #map1(%arg4)
      %10 = affine.apply #map(%arg5)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg6[%9, %10] [32, 16] [1, 1] : tensor<32x16xf32> into tensor<128x64xf32>
      }
    }
    return %0 : tensor<128x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %1 tile_sizes [32, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

