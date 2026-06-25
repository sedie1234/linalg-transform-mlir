// sched-a — match 로 모든 linalg structured op(fill+matmul x2)을 잡아
//   transform.foreach 로 각각에 generalize 를 적용한다.
//   match 한 번 -> foreach 가 payload op 하나씩 순회하며 body 실행.
//   결과: fill/matmul 3개 모두 linalg.generic 으로 풀린다.
module attributes {transform.with_named_sequence} {
  func.func @multi(%A1: tensor<64x128xf32>, %B1: tensor<128x64xf32>,
                   %A2: tensor<64x96xf32>,  %B2: tensor<96x64xf32>,
                   %init: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %0 = linalg.matmul ins(%A1, %B1 : tensor<64x128xf32>, tensor<128x64xf32>)
                       outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = linalg.matmul ins(%A2, %B2 : tensor<64x96xf32>, tensor<96x64xf32>)
                       outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // 모든 linalg structured op 매치 (fill + matmul x2 = 3개).
    %all = transform.structured.match interface{LinalgOp} in %arg0
      : (!transform.any_op) -> !transform.any_op
    // 매치된 각 op 에 대해 body 를 한 번씩 실행.
    transform.foreach %all : !transform.any_op {
    ^bb0(%op: !transform.any_op):
      transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op
    }
    transform.yield
  }
}
