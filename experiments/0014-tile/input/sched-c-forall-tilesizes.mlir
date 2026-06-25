// sched-c — tile_using_forall: tile_sizes [32, 32] -> 병렬 scf.forall.
// tile 크기를 직접 지정. thread 수는 ceildiv(dim, tile)로 유도 (M=4, N=2).
// num_threads(b) 와의 차이: forall iteration space가 num_threads 가 아니라
// tile_sizes 로부터 역산된다. result arity = (forall_op, tiled_op).
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
    %forall, %tiled = transform.structured.tile_using_forall %matmul tile_sizes [32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
