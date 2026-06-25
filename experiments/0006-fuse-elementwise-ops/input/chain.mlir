// #0006 positive 1 — FuseElementwiseOps 본체.
// elementwise linalg.generic 3-op 체인 (add → mul → sub).
// 각 producer 결과가 단일 사용이므로 defaultControlFn(hasOneUse) 통과,
// greedy 고정점까지 두 번 fuse 되어 generic 1개로 합쳐진다.
// (중간 tensor.empty 들은 dead 가 되어 greedy DCE 로 제거.)
#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_mul_sub_chain(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>,
                             %c: tensor<4x8xf32>, %d: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
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
      ins(%mul, %d : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.subf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>

  return %sub : tensor<4x8xf32>
}
