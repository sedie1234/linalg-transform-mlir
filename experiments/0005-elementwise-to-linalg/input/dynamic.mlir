// #0005 positive (2) — dynamic shape ranked tensor.
//
// 관찰 포인트: result type 이 operand 와 다른 cmpf 의 경우
// getOrCreateOperandsMatchingResultTypes (:67-70) 가 tensor.empty 를 만들 때
// tensor::getMixedSizes 로 *첫 operand* 에서 dynamic dim 을 추출 →
// dim 마다 tensor.dim (+ arith.constant index) 이 IR 에 추가된다.
// addf 쪽은 operand 재사용 분기라 tensor.dim 이 안 생기는 것과 대조.
func.func @dynamic_elementwise(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xi1>) {
  %sum = arith.addf %a, %b : tensor<?x?xf32>
  %cmp = arith.cmpf olt, %sum, %b : tensor<?x?xf32>
  return %sum, %cmp : tensor<?x?xf32>, tensor<?x?xi1>
}
