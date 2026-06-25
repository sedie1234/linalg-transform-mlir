// sched-b — tile_using_forall: num_threads [4, 2] -> 병렬 scf.forall.
// M축 4-way, N축 2-way 분할. result arity = (forall_op, tiled_op).
// num_threads 변형은 K축(reduction)을 타일링하지 않음 -> 단일 scf.forall.
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
    %forall, %tiled = transform.structured.tile_using_forall %matmul num_threads [4, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
