// sched-d — foreach + alternatives 결합.
//   match(matmul x2) -> foreach 로 각 matmul 을 순회.
//   각 matmul 에 transform.alternatives 적용:
//     Alt1: match.operation_name 이 "linalg.conv_2d_nhwc_hwcf" 인지 검사
//           -> 아니므로 silenceable failure -> 다음 alternative 로 폴백.
//     Alt2: match.operation_name 이 "linalg.matmul" 인지 검사 -> 성공.
//           이후 generalize 적용.
//   alternatives 는 IsolatedFromAbove 라 외부 핸들을 못 받음 -> scope(=각 matmul)
//   를 operand 로 주고, region arg(%scope)로 다시 매칭.
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
    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.foreach %mms : !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      transform.alternatives %mm : !transform.any_op {
      ^bb1(%scope: !transform.any_op):
        // Alt1: conv 인지 검사 -> matmul 이므로 실패(silenceable) -> 폴백.
        transform.match.operation_name %scope ["linalg.conv_2d_nhwc_hwcf"]
          : !transform.any_op
        transform.structured.generalize %scope : (!transform.any_op) -> !transform.any_op
      }, {
      ^bb1(%scope: !transform.any_op):
        // Alt2: matmul 인지 검사 -> 성공 -> generalize 적용.
        transform.match.operation_name %scope ["linalg.matmul"]
          : !transform.any_op
        transform.structured.generalize %scope : (!transform.any_op) -> !transform.any_op
      }
    }
    transform.yield
  }
}
