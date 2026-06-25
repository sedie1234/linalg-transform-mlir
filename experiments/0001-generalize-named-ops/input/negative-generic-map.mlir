// #0001 입력 3 (negative — pass 가 발화하지 않아야 함)
// generalizeNamedOpPrecondition (Generalization.cpp:38-51) 의 bail 분기 검증:
//   - linalg.generic : 이미 generic 이므로 isa<GenericOp> bail (:42)
//   - linalg.map     : block arg 구성이 generic 과 달라(output 이 region arg 에
//                      없음) trivially generalize 불가 → isa<MapOp> bail (:42)
// 출력이 입력과 동일(modulo 표준 재인쇄)해야 한다.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @negative(%x: tensor<4x16xf32>, %y: tensor<4x16xf32>,
                    %out: tensor<4x16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  // 이미 generic — 그대로 남아야 함
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %y : tensor<4x16xf32>, tensor<4x16xf32>)
      outs(%out : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32
  } -> tensor<4x16xf32>
  // linalg.map — 그대로 남아야 함 (MapOp bail)
  %1 = linalg.map { arith.addf } ins(%x, %y : tensor<4x16xf32>, tensor<4x16xf32>)
                                 outs(%out : tensor<4x16xf32>)
  return %0, %1 : tensor<4x16xf32>, tensor<4x16xf32>
}
