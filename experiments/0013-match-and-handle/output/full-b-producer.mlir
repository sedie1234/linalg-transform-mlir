[[[ IR printer: producer of matmul outs (expect linalg.fill) ]]]
%0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @mm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%arg2 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_producer_of_operand %0[2] : (!transform.any_op) -> !transform.any_op
    transform.print %1 {name = "producer of matmul outs (expect linalg.fill)"} : !transform.any_op
    %2 = transform.structured.generalize %1 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

