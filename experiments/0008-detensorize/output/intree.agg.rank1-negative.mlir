#map = affine_map<(d0) -> (d0)>
module {
  func.func @rank1_not_detensorable(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: i1) -> tensor<4xf32> {
    cf.cond_br %arg2, ^bb1, ^bb2(%arg1 : tensor<4xf32>)
  ^bb1:  // pred: ^bb0
    %0 = tensor.empty() : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    cf.br ^bb2(%1 : tensor<4xf32>)
  ^bb2(%2: tensor<4xf32>):  // 2 preds: ^bb0, ^bb1
    return %2 : tensor<4xf32>
  }
}

