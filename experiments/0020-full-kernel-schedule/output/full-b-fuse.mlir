#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module attributes {transform.with_named_sequence} {
  func.func @fc_relu(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (16, 4) shared_outs(%arg6 = %arg3) -> (tensor<128x128xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = affine.apply #map1(%arg5)
      %5 = affine.apply #map(%arg4)
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map(%arg4)
      %8 = affine.apply #map1(%arg5)
      %9 = affine.apply #map(%arg4)
      %10 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%7, 0] [8, 128] [1, 1] : tensor<128x128xf32> to tensor<8x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %8] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%9, %10] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %11 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x128xf32>, tensor<128x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%3, %4] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %extracted_slice_3 = tensor.extract_slice %arg3[%5, %6] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %extracted_slice_2 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_4 = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%12, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_4 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %14 = affine.apply #map(%arg4)
      %15 = affine.apply #map1(%arg5)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg6[%14, %15] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<128x128xf32>
      }
    }
    return %0 : tensor<128x128xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.split_handle %1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_op, %forall_op = transform.structured.tile_using_forall %2#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %0 into %new_containing_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

