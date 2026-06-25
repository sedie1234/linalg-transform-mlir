// Sched B — vectorize_children_and_apply_patterns (legacy/일괄 vectorizer).
// isolated-from-above target(여기선 func.func)이 필요 → get_parent_op {isolated_from_above}.
// func 안 모든 linalg child 를 vectorize + canonical vector pattern 적용.
// 결과: matmul -> vector.contract, transfer 패턴 정리됨.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<8x16xf32>, %B: tensor<16x4xf32>,
                    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<8x16xf32>, tensor<16x4xf32>)
                       outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %func = transform.get_parent_op %mm {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
