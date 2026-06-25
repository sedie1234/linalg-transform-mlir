// #0002 specialize-generic-ops — negative 입력 (어느 함수도 발화하지 않아야 함).
// specializeGenericOp 의 각 idiom 판정이 *어디서* reject 하는지를 1:1 로 본다:
//   @transpose_like   → maps 가 identity 가 아님 → isaCopy/isaElemwise 모두
//                       false. transpose 인식 분기는 pass 에 존재하지 않음
//                       (generalize 와의 round-trip 비대칭).
//   @fused_exp_neg    → body 가 2개 op (exp∘neg 융합) → isaElemwiseSingle*
//                       의 body size==2(op+yield) 검사에서 false (LinalgInterfaces.cpp:127-129).
//   @max_elemwise     → isaElemwiseSingleBinary 는 true 지만 arith.maximumf 는
//                       add/sub/mul/div 화이트리스트(:287-302) 밖 → failure.
//   @multi_k_contract → contraction 이긴 하나 reduction dim 2개 (k1,k2) →
//                       specializeLinalgContractions :182 `dims.k.size()!=1` reject.
#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#transpose2 = affine_map<(d0, d1) -> (d1, d0)>
#mkA = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#mkB = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
#mkC4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// yield 단독 body 지만 입력 map 이 (d1,d0) → copy idiom 탈락. 잔류.
func.func @transpose_like(%src: tensor<8x16xf32>, %init: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = linalg.generic
      {indexing_maps = [#transpose2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<8x16xf32>) outs(%init : tensor<16x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// producer-consumer 융합 산물 같은 2-op body → 단일 unary 아님. 잔류.
func.func @fused_exp_neg(%x: tensor<8x16xf32>, %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in : f32
    %e = math.exp %n : f32
    linalg.yield %e : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// 구조는 완벽한 elemwise single binary 지만 maximumf 분기가 없음. 잔류.
func.func @max_elemwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>,
                        %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.maximumf %in, %in_0 : f32
    linalg.yield %m : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// 유효한 linalg contraction 이지만 reduction 축이 {k1,k2} 2개 →
// named matmul 로 표현 불가 → reject. 잔류.
func.func @multi_k_contract(%A: tensor<10x20x30xf32>, %B: tensor<30x20x40xf32>,
                            %C: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.generic
      {indexing_maps = [#mkA, #mkB, #mkC4],
       iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%A, %B : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
      outs(%C : tensor<10x40xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}
