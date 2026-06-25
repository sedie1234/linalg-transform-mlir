// #0007 positive 입력 1 — DropUnitDims 본체 발화 케이스.
//
// DropUnitDims.cpp:174-229 의 파일 주석 예제를 실행 가능하게 옮긴 것.
// 두 입력이 unit dim (1x5, 5x1) 으로 broadcasting 을 표현하고, indexing map
// 이 그 자리를 상수 0 으로 접근한다. linalg::dropUnitDims 가
//   - operand 의 unit dim 을 tensor.collapse_shape 로 제거 (collapseValue)
//   - indexing map 을 (d0,d1)->(0,d1) → (d0,d1)->(d1) 로 재작성
//     (dropUnitExtentFromOperandMetadata: AffineConstantExpr(0) 자리 삭제)
// 한다. iteration dim 자체는 둘 다 5 라서 떨어지지 않는다 (unitDims 공집합
// 이어도 constant-0 접근 정리만으로 newIndexingMaps != indexingMaps 가 되어
// rewrite 가 일어나는 분기 — DropUnitDims.cpp:448-452 주석의 "legacy" 경로).
#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @broadcast_add(%arg0: tensor<1x5xf32>, %arg1: tensor<5x1xf32>)
    -> tensor<5x5xf32> {
  %empty = tensor.empty() : tensor<5x5xf32>
  %0 = linalg.generic
      {indexing_maps = [#map0, #map1, #map2],
       iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<1x5xf32>, tensor<5x1xf32>)
      outs(%empty : tensor<5x5xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// unit iteration dim 이 실제로 떨어지는 케이스: d0 (size 1) 가 one-trip.
// linalg.index 도 함께 있어 replaceUnitDimIndexOps (:232-251) 가
// index(0) → arith.constant 0, index(1) → index(0) 시프트를 수행한다.
#map3 = affine_map<(d0, d1) -> (d0, d1)>
func.func @drop_unit_loop_with_index(%arg0: tensor<1x8xf32>)
    -> tensor<1x8xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %0 = linalg.generic
      {indexing_maps = [#map3, #map3],
       iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<1x8xf32>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %sum = arith.addi %i, %j : index
    %cast = arith.index_cast %sum : index to i32
    %f = arith.sitofp %cast : i32 to f32
    %add = arith.addf %in, %f : f32
    linalg.yield %add : f32
  } -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
