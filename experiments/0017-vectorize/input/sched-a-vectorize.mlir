// Sched A — transform.structured.vectorize (정밀 vectorizer).
// linalg.matmul 을 직접 매치해 vectorize. result type 없음(in-place, op 소비 X).
// 결과: vector.transfer_read + vector.contract / multi_reduction + vector.transfer_write.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<8x16xf32>, %B: tensor<16x4xf32>,
                    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<8x16xf32>, tensor<16x4xf32>)
                       outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %mm : !transform.any_op
    transform.yield
  }
}
