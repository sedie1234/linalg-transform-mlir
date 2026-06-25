// T04 payload — linalg.matmul (128x256) x (256x64) -> (128x64).
// 변환 대상만. transform script는 sched-*.mlir 에 별도.
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>,
                  %C: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                     outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
