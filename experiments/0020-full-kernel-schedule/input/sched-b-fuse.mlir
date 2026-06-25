// T10 step B — match → tile → fuse. Tile relu(max) into scf.forall, then
//   fuse the producers (add, then matmul) into the same loop with
//   fuse_into_containing_op. Result: one forall nest with matmul+add+max
//   on tiles (producer-consumer fusion).
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

    // Tile the consumer (relu/max) into a forall loop.
    %tiled, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse producers into the loop, consumer-before-producer order.
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm_fused, %loop3 = transform.structured.fuse_into_containing_op %mm into %loop2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
