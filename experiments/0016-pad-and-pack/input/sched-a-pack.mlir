// sched-a — 기본 transform.structured.pack.
// matmul (M,N,K) 차원을 packed_sizes=[4,8,8]로 block-pack.
// 효과: matmul → tensor.pack ×3 (A,B,C) + linalg.generic(6D) + tensor.unpack ×1.
// 0010 block-pack 패스가 IR에서 한 일을 schedule IR로 명시한 버전.
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
    //                                              M  N  K
    %packed = transform.structured.pack %matmul packed_sizes = [4, 8, 8]
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}
