#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @multi(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64x96xf32>, %arg3: tensor<96x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<64x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg2, %arg3 : tensor<64x96xf32>, tensor<96x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<64x64xf32>
    return %1, %2 : tensor<64x64xf32>, tensor<64x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.alternatives %0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.match.operation_name %1 ["linalg.matmul"] : !transform.any_op
      %2 = transform.structured.generalize %1 : (!transform.any_op) -> !transform.any_op
    }, {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.foreach %1 : !transform.any_op {
      ^bb0(%arg2: !transform.any_op):
        %2 = transform.structured.generalize %arg2 : (!transform.any_op) -> !transform.any_op
      }
    }
    transform.yield 
  }
}

