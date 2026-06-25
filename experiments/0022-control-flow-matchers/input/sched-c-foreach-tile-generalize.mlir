// sched-c — foreach body 안에서 두 변환을 연쇄: tile -> generalize.
//   match(matmul x2) -> foreach { tile_using_for; generalize(tiled) }.
//   foreach body 의 핸들(%tiled)이 같은 body 안에서 다음 op 로 흘러감을 보임.
//   결과: 각 matmul 이 tile 된 뒤, 안쪽 32x32 matmul 이 linalg.generic 으로 풀림.
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
      // 1) 각 matmul 을 [32,32,32] tile.
      %tiled, %loops:3 = transform.structured.tile_using_for %mm tile_sizes [32, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      // 2) tile 된 안쪽 matmul 을 generic 으로 generalize.
      transform.structured.generalize %tiled : (!transform.any_op) -> !transform.any_op
    }
    transform.yield
  }
}
