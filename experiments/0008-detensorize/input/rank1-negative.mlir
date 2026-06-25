// 진짜 negative: rank-1 tensor<4xf32> 는 canBeDetensored()=false
// (Detensorize.cpp:51-53 — rank 0 만 detensor 가능) 이므로
// DetensorizeTypeConverter 가 type 을 그대로 두고(= legal),
// shouldBeDetensored() 도 false → 기본/aggressive 어느 모드에서도 발화 X.
// 제어흐름(cond_br)에 tensor 가 관여해도 rank>0 이면 대상이 아님을 보인다.
#map1 = affine_map<(d0) -> (d0)>

func.func @rank1_not_detensorable(%a: tensor<4xf32>, %b: tensor<4xf32>,
                                  %cond: i1) -> tensor<4xf32> {
  cf.cond_br %cond, ^bb1(%a : tensor<4xf32>), ^bb2(%b : tensor<4xf32>)

^bb1(%x: tensor<4xf32>):
  %init = tensor.empty() : tensor<4xf32>
  %sum = linalg.generic
           {indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel"]}
           ins(%x, %x : tensor<4xf32>, tensor<4xf32>)
           outs(%init : tensor<4xf32>) {
    ^bb0(%e0: f32, %e1: f32, %out: f32):
      %s = arith.addf %e0, %e1 : f32
      linalg.yield %s : f32
  } -> tensor<4xf32>
  cf.br ^bb2(%sum : tensor<4xf32>)

^bb2(%r: tensor<4xf32>):
  return %r : tensor<4xf32>
}
