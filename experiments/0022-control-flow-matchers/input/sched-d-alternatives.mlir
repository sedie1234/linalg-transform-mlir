// sched-d — transform.alternatives (제어흐름: 첫 성공 분기 채택).
//   alternatives 는 IsolatedFromAbove scope 만 받음 -> func.func 를 scope 로.
//   match(func.func) -> alternatives:
//     Alt1: fill op 하나를 잡아 match.operation_name 으로 "linalg.matmul" 인지 검사
//           -> fill 이므로 silenceable failure -> Alt1 전체 폐기, Alt2 로 폴백.
//           (Alt1 의 generalize 는 실행되지 않으므로 부작용 없음.)
//     Alt2: matmul 2개를 잡아 foreach 로 각각 generalize -> 성공.
//   즉 "조건 검사 실패 시 다른 변환 경로" 를 보인다. foreach 와도 결합.
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
    // alternatives 의 scope 로 쓸 func.func 를 매치.
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.alternatives %func : !transform.any_op {
    ^bb1(%scope: !transform.any_op):
      // Alt1: fill 을 잡아 "matmul 인가?" 검사 -> 거짓 -> silenceable fail -> 폴백.
      %fill = transform.structured.match ops{["linalg.fill"]} in %scope
        : (!transform.any_op) -> !transform.any_op
      transform.match.operation_name %fill ["linalg.matmul"] : !transform.any_op
      // (도달 불가) 만약 통과했다면 fill 을 generalize 했을 것.
      transform.structured.generalize %fill : (!transform.any_op) -> !transform.any_op
    }, {
    ^bb1(%scope: !transform.any_op):
      // Alt2: matmul 2개를 잡아 foreach 로 각각 generalize -> 성공.
      %mms = transform.structured.match ops{["linalg.matmul"]} in %scope
        : (!transform.any_op) -> !transform.any_op
      transform.foreach %mms : !transform.any_op {
      ^bb2(%mm: !transform.any_op):
        transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
      }
    }
    transform.yield
  }
}
