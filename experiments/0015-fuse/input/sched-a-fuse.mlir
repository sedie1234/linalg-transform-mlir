// Variant A: transform.structured.fuse (tile + fuse) on the matmul root.
//
// fuse tiles the matched root (matmul) with the given tile_sizes, creating an
// scf.for nest, then greedily fuses its tileable producers (the elemwise_binary
// add) into that nest. Result: producer's tiled slice lives inside the loops.
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
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // tile the two parallel dims (M, N); K dim tile = 0 -> not tiled here.
    %fused, %loops:2 = transform.structured.fuse %mm
        {tile_sizes = [32, 32], tile_interchange = [0, 1]}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
