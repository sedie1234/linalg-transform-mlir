// #0005 positive (1) — static shape ranked tensor 위의 ElementwiseMappable op 들.
//
// 발화 대상 (모든 operand 가 RankedTensorType + 4 trait):
//   - arith.addf / arith.mulf : result type == operand type → DPS init 으로
//     기존 operand 재사용 (getOrCreateOperandsMatchingResultTypes 의 found 분기)
//   - math.exp                : arith 외 dialect 도 trait 기반으로 잡힘
//   - arith.cmpf              : result tensor<8x16xi1> 은 어떤 operand type 과도
//     달라 tensor.empty 가 새로 생성되는 분기. predicate attr 이 body 의
//     scalar cmpf 로 그대로 전달되는지 관찰 (op->getAttrs() 전달 :107)
//   - arith.select            : 3-operand (i1 tensor + f32 tensor 2개) —
//     indexing map 4개(3 in + 1 out) 전부 identity
func.func @static_elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
    -> (tensor<8x16xf32>, tensor<8x16xi1>, tensor<8x16xf32>) {
  %sum = arith.addf %a, %b : tensor<8x16xf32>
  %prod = arith.mulf %sum, %a : tensor<8x16xf32>
  %exp = math.exp %prod : tensor<8x16xf32>
  %cmp = arith.cmpf ogt, %a, %b : tensor<8x16xf32>
  %sel = arith.select %cmp, %exp, %b : tensor<8x16xi1>, tensor<8x16xf32>
  return %exp, %cmp, %sel : tensor<8x16xf32>, tensor<8x16xi1>, tensor<8x16xf32>
}
