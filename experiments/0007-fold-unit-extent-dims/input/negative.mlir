// #0007 negative 입력 — 어떤 패턴도 발화하지 않는 케이스.
//
// unit-extent dim 이 전혀 없다: 모든 shape 가 4x8, indexing map 은 identity,
// AffineConstantExpr(0) 접근도 없다. linalg::dropUnitDims 는
// newIndexingMaps == indexingMaps 라서 failure (DropUnitDims.cpp:489-493),
// DropPadUnitDims/RankReduced*SliceOp 는 대상 op 자체가 없고,
// MoveInitOperandsToInput 은 body 가 %out 을 읽지 않으므로 candidates 가
// 비어 failure (:100-101). 출력은 입력과 동일해야 한다.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_unit_dims(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%empty : tensor<4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    linalg.yield %mul : f32
  } -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
