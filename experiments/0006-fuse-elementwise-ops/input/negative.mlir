// #0006 negative — fusion 이 발화하지 않는 두 가지 사유.
// (1) @multi_use_producer: producer %add 결과를 consumer 2개가 사용 →
//     합법성(areElementwiseOpsFusable)은 통과하지만 defaultControlFn
//     (producer->hasOneUse(), ElementwiseOpFusion.cpp:2139-2142) 이 차단.
// (2) @reduction_then_elementwise: producer 가 reduction iterator 보유 →
//     areElementwiseOpsFusable 의 all-parallel 검사
//     (ElementwiseOpFusion.cpp:113-114) 에서 탈락.
// 어느 패턴도 발화하지 않아 출력 = 입력 (round-trip 포맷 차이만).
#map = affine_map<(d0, d1) -> (d0, d1)>
#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_red = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0) -> (d0)>

func.func @multi_use_producer(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>,
                              %c: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %e0 = tensor.empty() : tensor<4x8xf32>
  %add = linalg.generic
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%e0 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>

  %e1 = tensor.empty() : tensor<4x8xf32>
  %mul = linalg.generic
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}
      ins(%add, %c : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%e1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>

  %e2 = tensor.empty() : tensor<4x8xf32>
  %sub = linalg.generic
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}
      ins(%add, %c : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.subf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>

  return %mul, %sub : tensor<4x8xf32>, tensor<4x8xf32>
}

func.func @reduction_then_elementwise(%a: tensor<4x8xf32>, %c: tensor<4xf32>)
    -> tensor<4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%e0 : tensor<4xf32>)
      -> tensor<4xf32>
  // producer: row-sum (d1 이 reduction) — all-parallel 아님.
  %sum = linalg.generic
      {indexing_maps = [#map_in, #map_red],
       iterator_types = ["parallel", "reduction"]}
      ins(%a : tensor<4x8xf32>) outs(%fill : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %0 = arith.addf %acc, %x : f32
    linalg.yield %0 : f32
  } -> tensor<4xf32>

  %e1 = tensor.empty() : tensor<4xf32>
  %r = linalg.generic
      {indexing_maps = [#map1, #map1, #map1],
       iterator_types = ["parallel"]}
      ins(%sum, %c : tensor<4xf32>, tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
