// 발화 케이스 (b): 상수 인덱스 접근 operand.
// %table 의 indexing map 은 (d0, d1) -> (0, 1) — 모든 result 가
// AffineConstantExpr 이므로 isConstant() == true.
// body 선두에 %c0/%c1 = arith.constant {0,1} : index 와
// tensor.extract %table[%c0, %c1] 이 생성되어 inline 된다.
#map_const = affine_map<(d0, d1) -> (0, 1)>
#map_id    = affine_map<(d0, d1) -> (d0, d1)>

func.func @inline_const_index(%table: tensor<2x3xf32>, %x: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %init = tensor.empty() : tensor<4x5xf32>
  %res = linalg.generic
      {indexing_maps = [#map_const, #map_id, #map_id],
       iterator_types = ["parallel", "parallel"]}
      ins(%table, %x : tensor<2x3xf32>, tensor<4x5xf32>)
      outs(%init : tensor<4x5xf32>) {
  ^bb0(%t: f32, %xi: f32, %o: f32):
    %m = arith.mulf %t, %xi : f32
    linalg.yield %m : f32
  } -> tensor<4x5xf32>
  return %res : tensor<4x5xf32>
}
