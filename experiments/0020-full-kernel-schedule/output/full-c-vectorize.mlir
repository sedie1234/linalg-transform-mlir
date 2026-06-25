#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @fc_relu(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<8x32xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (16, 4) shared_outs(%arg6 = %arg3) -> (tensor<128x128xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = vector.transfer_read %arg0[%3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<8x128xf32>
      %5 = affine.apply #map1(%arg5)
      %6 = vector.transfer_read %arg1[%c0, %5], %cst_0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x32xf32>
      %7 = affine.apply #map(%arg4)
      %8 = affine.apply #map1(%arg5)
      %9 = vector.transfer_read %arg3[%7, %8], %cst_0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<8x32xf32>
      %10 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %6, %9 : vector<8x128xf32>, vector<128x32xf32> into vector<8x32xf32>
      %11 = affine.apply #map(%arg4)
      %12 = affine.apply #map1(%arg5)
      %13 = vector.transfer_read %arg2[%11, %12], %cst_0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<8x32xf32>
      %14 = arith.addf %10, %13 : vector<8x32xf32>
      %extracted_slice = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<128x128xf32> to tensor<8x32xf32>
      %15 = arith.maximumf %14, %cst : vector<8x32xf32>
      %16 = vector.transfer_write %15, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<8x32xf32>
      %17 = affine.apply #map(%arg4)
      %18 = affine.apply #map1(%arg5)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg6[%17, %18] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<128x128xf32>
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
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.vectorize_children_and_apply_patterns %3 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

