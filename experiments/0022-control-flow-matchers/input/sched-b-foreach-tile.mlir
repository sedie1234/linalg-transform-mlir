// sched-b — ops{["linalg.matmul"]} 로 matmul 2개만 매치, foreach 로 각각 tile.
//   fill 은 매치 대상 아님 -> 손대지 않음.
//   foreach body 안에서 tile_using_for -> 각 matmul 이 자기만의 scf.for 중첩을 가짐.
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
    // matmul 만 매치 (2개). fill 제외.
    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    // 각 matmul 을 [32,32,32] 로 독립적으로 tile.
    transform.foreach %mms : !transform.any_op {
    ^bb0(%mm: !transform.any_op):
      %tiled, %loops:3 = transform.structured.tile_using_for %mm tile_sizes [32, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    transform.yield
  }
}
