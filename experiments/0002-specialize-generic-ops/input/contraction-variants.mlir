// #0002 specialize-generic-ops — contraction idiom 입력.
// isaContractionOpInterface (:305) → specializeLinalgContractions
// (Specialize.cpp:148-255) 의 matmul 변형 판정을 발화시킨다:
//   @matmul_f32          → A/B/C 모두 Match            → linalg.matmul
//   @matmul_transpose_b  → B 가 Transposed             → linalg.matmul_transpose_b
//   @matmul_transpose_a  → A 가 Transposed             → linalg.matmul_transpose_a
//   @batch_matmul_i32    → batch dim + muli/addi body  → linalg.batch_matmul
#mmA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mmB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mmC = affine_map<(d0, d1, d2) -> (d0, d1)>
#mmAT = affine_map<(d0, d1, d2) -> (d2, d0)>
#mmBT = affine_map<(d0, d1, d2) -> (d1, d2)>
#bmA = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bmB = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#bmC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// 표준 matmul: A(m,k) B(k,n) C(m,n), body = mulf+addf.
// matchOperandMap(A)=Match, (B)=Match, (C)=Match → MatmulOp (:254).
func.func @matmul_f32(%A: tensor<16x8xf32>, %B: tensor<8x32xf32>,
                      %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
      {indexing_maps = [#mmA, #mmB, #mmC],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<16x8xf32>, tensor<8x32xf32>)
      outs(%C : tensor<16x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// B 의 map 이 (d1,d2) = (n,k) → matchOperandMap(B)=Transposed →
// MatmulTransposeBOp (:253).
func.func @matmul_transpose_b(%A: tensor<16x8xf32>, %B: tensor<32x8xf32>,
                              %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
      {indexing_maps = [#mmA, #mmBT, #mmC],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<16x8xf32>, tensor<32x8xf32>)
      outs(%C : tensor<16x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// A 의 map 이 (d2,d0) = (k,m) → matchOperandMap(A)=Transposed →
// MatmulTransposeAOp (:251).
func.func @matmul_transpose_a(%A: tensor<8x16xf32>, %B: tensor<8x32xf32>,
                              %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic
      {indexing_maps = [#mmAT, #mmB, #mmC],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<8x16xf32>, tensor<8x32xf32>)
      outs(%C : tensor<16x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// 선두 batch dim (d0, 모든 map 에서 identity 위치) + muli/addi (정수 계열 페어)
// → numOfBatchDims=1 → BatchMatmulOp (:247).
func.func @batch_matmul_i32(%A: tensor<4x16x8xi32>, %B: tensor<4x8x32xi32>,
                            %C: tensor<4x16x32xi32>) -> tensor<4x16x32xi32> {
  %0 = linalg.generic
      {indexing_maps = [#bmA, #bmB, #bmC],
       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<4x16x8xi32>, tensor<4x8x32xi32>)
      outs(%C : tensor<4x16x32xi32>) {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %m = arith.muli %a, %b : i32
    %s = arith.addi %c, %m : i32
    linalg.yield %s : i32
  } -> tensor<4x16x32xi32>
  return %0 : tensor<4x16x32xi32>
}
