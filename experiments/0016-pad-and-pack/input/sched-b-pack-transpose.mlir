// sched-b — pack 후 pack_transpose.
// B operand(operand index 1)의 tensor.pack을 잡아 outer/inner perm=[1,0]로
// 레이아웃을 뒤집는다. KN-packing을 NK-packing으로 바꾸는 효과 (B를 transpose해
// inner tile 순서를 hardware-friendly 하게). compute_op의 indexing_map도 함께 갱신됨.
func.func @matmul(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>,
                  %C: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%C : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %matmul packed_sizes = [4, 8, 8]
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    // operand 1 = B 의 producer pack 을 집어온다.
    %pack_b = transform.get_producer_of_operand %packed[1]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"tensor.pack">)
    %generic2, %pack2, %unpack2 =
      transform.structured.pack_transpose %pack_b with_compute_op(%packed)
      outer_perm = [1, 0] inner_perm = [1, 0]
      : (!transform.op<"tensor.pack">, !transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.op<"tensor.pack">, !transform.any_op)
    transform.yield
  }
}
