module attributes {transform.with_named_sequence} {
  func.func @copy_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.copy ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}

