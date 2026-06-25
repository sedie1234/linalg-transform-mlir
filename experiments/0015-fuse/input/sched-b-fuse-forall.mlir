// Variant B: explicit two-step fuse via scf.forall container.
//
//   1) tile_using_forall on the matmul root  -> creates an scf.forall container,
//      matmul body operates on a tile.
//   2) fuse_into_containing_op pulls the elemwise_binary producer into that
//      forall, tiling/cloning it so its slice is computed inside the container.
//
// This is the lower-level decomposition of what `fuse` does in one shot, but
// with an scf.forall (parallel) container instead of scf.for.
module attributes {transform.with_named_sequence} {
  func.func @mm_chain(%A: tensor<128x256xf32>, %A2: tensor<128x256xf32>,
                      %B: tensor<256x64xf32>, %C: tensor<128x64xf32>)
      -> tensor<128x64xf32> {
    %AA = linalg.elemwise_binary
        ins(%A, %A2 : tensor<128x256xf32>, tensor<128x256xf32>)
        outs(%A : tensor<128x256xf32>) -> tensor<128x256xf32>
    %0 = linalg.matmul
        ins(%AA, %B : tensor<128x256xf32>, tensor<256x64xf32>)
        outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %prod = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %root = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // 1) tile root into an scf.forall container (tile M=32, N=16).
    %tiled, %forall = transform.structured.tile_using_forall %root tile_sizes [32, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // 2) fuse the elemwise_binary producer into that container.
    %fused, %new_container = transform.structured.fuse_into_containing_op %prod into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
