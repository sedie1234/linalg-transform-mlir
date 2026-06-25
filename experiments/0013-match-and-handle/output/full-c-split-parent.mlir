[[[ IR printer: both generics (single handle, 2 payloads) ]]]
%1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %4 = arith.addf %in, %in_0 : f32
  linalg.yield %4 : f32
} -> tensor<32xf32>
%3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %4 = arith.mulf %in, %in_0 : f32
  linalg.yield %4 : f32
} -> tensor<32xf32>
[[[ IR printer: split #0 (expect addf generic) ]]]
%1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %4 = arith.addf %in, %in_0 : f32
  linalg.yield %4 : f32
} -> tensor<32xf32>
[[[ IR printer: split #1 (expect mulf generic) ]]]
%3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %4 = arith.mulf %in, %in_0 : f32
  linalg.yield %4 : f32
} -> tensor<32xf32>
[[[ IR printer: get_parent_op of split #1 (expect func.func @two_gen) ]]]
func.func @two_gen(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<32xf32>
  %2 = tensor.empty() : tensor<32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.mulf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<32xf32>
  return %3 : tensor<32xf32>
}
#map = affine_map<(d0) -> (d0)>
module attributes {transform.with_named_sequence} {
  func.func @two_gen(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<32xf32>
    %2 = tensor.empty() : tensor<32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<32xf32>
    return %3 : tensor<32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.print %0 {name = "both generics (single handle, 2 payloads)"} : !transform.any_op
    %1:2 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %1#0 {name = "split #0 (expect addf generic)"} : !transform.any_op
    transform.print %1#1 {name = "split #1 (expect mulf generic)"} : !transform.any_op
    %2 = transform.get_parent_op %1#1 {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    transform.print %2 {name = "get_parent_op of split #1 (expect func.func @two_gen)"} : !transform.any_op
    transform.yield 
  }
}

