// sched-c — pack_greedily 로 자동 block-pack 후, 생성된 tensor.pack 하나를
// lower_pack 으로 분해. tensor.pack 1개는 padding이 필요 없으면
// tensor.expand_shape + linalg.transpose 로, padding이 필요하면 tensor.pad 가
// 추가로 붙어 lower 된다. 여기서는 pad+expand_shape+transpose 세 갈래 결과 타입.
func.func @matmul_mk_kn_mn(%A: tensor<1023x255xf32>, %B: tensor<255x127xf32>,
                           %C: tensor<1023x127xf32>) -> tensor<1023x127xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<1023x255xf32>, tensor<255x127xf32>)
                     outs(%C : tensor<1023x127xf32>) -> tensor<1023x127xf32>
  return %0 : tensor<1023x127xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.matmul">
    // greedily: heuristic으로 M,N,K inner-tile을 잡아 generic으로 pack.
    %generic = transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.matmul">) -> !transform.op<"linalg.generic">
    // 생성된 tensor.pack 들 중 첫 번째(A operand)를 lower.
    %pack = transform.get_producer_of_operand %generic[0]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"tensor.pack">)
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    transform.yield
  }
}
