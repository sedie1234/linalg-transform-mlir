#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module attributes {transform.with_named_sequence} {
  func.func @fc_relu(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = scf.forall (%arg4, %arg5) in (16, 4) shared_outs(%arg6 = %arg3) -> (tensor<128x128xf32>) {
      %3 = affine.apply #map(%arg4)
      %4 = affine.apply #map1(%arg5)
      %5 = affine.apply #map(%arg4)
      %6 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %1[%3, %4] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg6[%5, %6] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%extracted_slice, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_0 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %8 = affine.apply #map(%arg4)
      %9 = affine.apply #map1(%arg5)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %arg6[%8, %9] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<128x128xf32>
      }
    }
    return %2 : tensor<128x128xf32>
  }
}

