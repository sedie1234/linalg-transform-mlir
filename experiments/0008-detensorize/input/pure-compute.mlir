// 제어흐름에 관여하지 *않는* 0-d tensor linalg.generic (in-tree test
// detensorize_0d.mlir 성격). generic 의 결과가 br/cond_br operand 로 흐르지
// 않고 곧장 return 된다.
// → 기본 모드(ControlFlowDetectionModel): cond_br/br 가 없어 workList 가
//   비어 있으므로 opsToDetensor = {} → 발화하지 않음 (negative).
// → aggressive-mode(AggressiveDetensoringModel): walk 로 모든
//   shouldBeDetensored generic 을 수집 → 발화 (positive).
#map = affine_map<() -> ()>

func.func @detensorable_but_no_cf(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %init = tensor.empty() : tensor<f32>
  %res = linalg.generic
           {indexing_maps = [#map, #map, #map], iterator_types = []}
           ins(%a, %b : tensor<f32>, tensor<f32>)
           outs(%init : tensor<f32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
  } -> tensor<f32>
  return %res : tensor<f32>
}
