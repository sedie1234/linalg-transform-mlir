// #0005 negative — 발화하지 않는 케이스 3종. 전부 불변이어야 한다.
//
// (1) scalar operand: ElementwiseMappable trait 은 있으나 operand 가
//     RankedTensorType 이 아님 (f32) → 술어의 all_of 실패.
// (2) vector operand: 마찬가지로 RankedTensorType 아님 (vector<4xf32>).
//     (Vectorizable trait 이 있어도 이 pass 의 대상은 tensor 뿐.)
// (3) scalar condition 의 arith.select: %cond 가 i1 *scalar* 라
//     "모든 operand 가 ranked tensor" (all_of, ElementwiseToLinalg.cpp:30)
//     를 깬다 — 소스 :28-29 의 TODO ("any_of 로 일반화 가능하나 미구현")
//     가 말하는 바로 그 케이스.
func.func @scalar_add(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

func.func @vector_add(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.addf %a, %b : vector<4xf32>
  return %0 : vector<4xf32>
}

func.func @select_scalar_cond(%cond: i1, %a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %0 = arith.select %cond, %a, %b : tensor<8xf32>
  return %0 : tensor<8xf32>
}
