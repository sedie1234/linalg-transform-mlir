// #0010 positive 1 — 정확히 나눠떨어지는 named linalg.matmul.
// M=64, K=256, N=128.  block-factors=32,16,64 (mb,nb,kb) 와 조합하면
//   M/mb=2, N/nb=8, K/kb=4 — 전부 정수이므로 padding 없는 깨끗한 packing.
// BlockPackMatmul<linalg::MatmulOp> (BlockPackMatmul.cpp:217-234, primary
// template) 가 발화하는 경로.
func.func @matmul_f32(%A: tensor<64x256xf32>, %B: tensor<256x128xf32>,
                      %C: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x256xf32>, tensor<256x128xf32>)
                     outs(%C : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
