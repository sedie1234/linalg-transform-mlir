// Variant C: transform.structured.fuse on the elemwise->elemwise chain.
//
// Match the elemwise_binary consumer as root, tile it, and fuse the
// elemwise_unary producer into the resulting scf.for nest. Mirrors the canonical
// upstream fuse test (@fuse_unary) but on a static shape so the IR is concrete.
module attributes {transform.with_named_sequence} {
  func.func @elem_chain(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>)
      -> tensor<512x512xf32> {
    %0 = linalg.elemwise_unary ins(%arg0 : tensor<512x512xf32>)
                               outs(%arg1 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<512x512xf32>, tensor<512x512xf32>)
                                outs(%arg1 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %1 : tensor<512x512xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %root = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused, %loops:2 = transform.structured.fuse %root
        {tile_sizes = [64, 64], tile_interchange = [0, 1]}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
