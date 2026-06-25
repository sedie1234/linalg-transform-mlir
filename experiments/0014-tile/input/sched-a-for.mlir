// sched-a — tile_using_for: tile_sizes [32, 32, 64] -> 순차 scf.for 3중 nest.
// M=128/32=4, N=64/32=2, K=256/64=4. result arity = (tiled_op, loop_M, loop_N, loop_K).
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>,
                    %C: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                       outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
