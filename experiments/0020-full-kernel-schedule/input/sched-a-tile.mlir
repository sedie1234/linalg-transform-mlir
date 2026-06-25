// T10 step A — match → tile (forall). Tile only the final relu (max) op.
//   match matmul + both elemwise_binary, split_handle to isolate add/max,
//   tile_using_forall on max. No fusion yet — baseline single-op tile.
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
    %ew = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %add, %max = transform.split_handle %ew
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
