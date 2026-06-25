// T06 payload — 32x32 square matmul. pack/pack_transpose가 padding 없이
// 깔끔히 떨어지도록 32를 packed_size(4,8,8)의 배수로 잡았다.
func.func @matmul(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>,
                  %C: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%C : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
