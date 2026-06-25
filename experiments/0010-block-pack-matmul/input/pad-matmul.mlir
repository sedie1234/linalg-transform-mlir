// #0010 positive 3 / conditional-negative — 나눠떨어지지 않는 크기.
// M=N=K=30, block-factors=16,16,16 → 30 % 16 != 0.
//   allow-padding=true (기본): linalg::pack (Transforms.cpp:480-610) 이
//     padding_value(f32 0.0) 붙은 tensor.pack 생성 → 2x2x16x16 blocked.
//   allow-padding=false: blockPackMatmul (BlockPackMatmul.cpp:154-159) 의
//     validateFullTilesOnDims (:44-86) 가 30 % 16 != 0 으로 실패 →
//     "expect packing full tiles only" matchFailure → no-op.
func.func @pad_matmul(%A: tensor<30x30xf32>, %B: tensor<30x30xf32>,
                      %C: tensor<30x30xf32>) -> tensor<30x30xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<30x30xf32>, tensor<30x30xf32>)
                     outs(%C : tensor<30x30xf32>) -> tensor<30x30xf32>
  return %0 : tensor<30x30xf32>
}
