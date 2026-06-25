// #0010 positive 2 — linalg.generic 으로 쓴 matmul_transpose_b 형 contraction.
// indexing maps = {(i,k), (j,k), (i,j)} — BlockPackMatmul<linalg::GenericOp>
// 전문화 (BlockPackMatmul.cpp:236-276) 가 허용하는 세 형태(:261-263) 중
// 세 번째.  RHS 가 이미 [N][K] (transposed) layout 이므로
// transposePackedMatmul 의 isOuterTransposed/isInnerTransposed 판정
// (:107-110) 이 기본 rhs-transpose-*=true 와 이미 일치 → RHS 재transpose 가
// 필요 없는 경로를 관찰한다.
// M=N=K=32, block-factors=8,8,8 → 나눠떨어짐.
#map_a = affine_map<(i, j, k) -> (i, k)>
#map_b = affine_map<(i, j, k) -> (j, k)>
#map_c = affine_map<(i, j, k) -> (i, j)>
func.func @generic_transpose_b(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>,
                               %C: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map_a, #map_b, #map_c],
          iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<32x32xf32>, tensor<32x32xf32>)
         outs(%C : tensor<32x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
