module attributes {transform.with_named_sequence} {
  func.func @copy_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.copy ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

