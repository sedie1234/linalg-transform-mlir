// #0006 positive 2 — 보조 패턴 2종.
// (1) @collapse_into_generic: input 의 tensor.collapse_shape 를
//     FoldWithProducerReshapeOpByExpansion 이 generic 안으로 흡수 —
//     generic 의 iteration space 가 2D→3D 로 확장되고, 다른 operand 에
//     expand_shape 가, 결과에 collapse_shape 가 생긴다 (reshape 가
//     generic 아래로 전파).
// (2) @splat_fold: splat arith.constant input 을 FoldScalarOrSplatConstant
//     가 body 안 스칼라 상수로 끌어들여 operand 에서 제거.
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @collapse_into_generic(%a: tensor<2x3x4xf32>, %b: tensor<6x4xf32>)
    -> tensor<6x4xf32> {
  %collapsed = tensor.collapse_shape %a [[0, 1], [2]]
      : tensor<2x3x4xf32> into tensor<6x4xf32>
  %e = tensor.empty() : tensor<6x4xf32>
  %r = linalg.generic
      {indexing_maps = [#map2, #map2, #map2],
       iterator_types = ["parallel", "parallel"]}
      ins(%collapsed, %b : tensor<6x4xf32>, tensor<6x4xf32>)
      outs(%e : tensor<6x4xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<6x4xf32>
  return %r : tensor<6x4xf32>
}

func.func @splat_fold(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst = arith.constant dense<2.000000e+00> : tensor<4x8xf32>
  %e = tensor.empty() : tensor<4x8xf32>
  %r = linalg.generic
      {indexing_maps = [#map2, #map2, #map2],
       iterator_types = ["parallel", "parallel"]}
      ins(%a, %cst : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
