#map = affine_map<(d0) -> (d0 * 32)>
module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = scf.forall (%arg3, %arg4) in (4, 2) shared_outs(%arg5 = %arg2) -> (tensor<128x64xf32>) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map(%arg4)
      %3 = affine.apply #map(%arg3)
      %4 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%3, %4] [32, 32] [1, 1] : tensor<128x64xf32> to tensor<32x32xf32>
      %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %6 = affine.apply #map(%arg3)
      %7 = affine.apply #map(%arg4)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg5[%6, %7] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
      }
    }
    return %0 : tensor<128x64xf32>
  }
}

