// T12 payload — 여러 linalg op 혼합.
//   fill (init) + matmul1 + matmul2.
// transform script가 match -> foreach 로 각 op에 변환을 적용하는 대상.
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
