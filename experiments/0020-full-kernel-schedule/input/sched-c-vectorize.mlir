// T10 step C — full kernel schedule: match → tile → fuse → vectorize.
//   Same tile+fuse as B, then vectorize the whole function body via
//   vectorize_children_and_apply_patterns on the isolated-from-above parent
//   (func.func). Linalg ops on tiles become vector.transfer_read/contract/
//   transfer_write; the codegen lowering target.
module attributes {transform.with_named_sequence} {
  func.func @fc_relu(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>,
                     %bias: tensor<128x128xf32>, %output: tensor<128x128xf32>)
                     -> tensor<128x128xf32> {
    %matmul = linalg.matmul ins(%lhs, %rhs: tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%output: tensor<128x128xf32>) -> tensor<128x128xf32>
    %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
      ins(%matmul, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%output : tensor<128x128xf32>) -> tensor<128x128xf32>
    %c0f = arith.constant 0.0 : f32
    %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
      ins(%biased, %c0f : tensor<128x128xf32>, f32)
      outs(%output : tensor<128x128xf32>) -> tensor<128x128xf32>
    func.return %relued : tensor<128x128xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %ew = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %add, %max = transform.split_handle %ew
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile the consumer (relu/max), then fuse producers in.
    %tiled, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm_fused, %loop3 = transform.structured.fuse_into_containing_op %mm into %loop2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Vectorize: grab the isolated-from-above parent (func.func) and vectorize
    // all linalg children, applying canonicalization patterns.
    %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
