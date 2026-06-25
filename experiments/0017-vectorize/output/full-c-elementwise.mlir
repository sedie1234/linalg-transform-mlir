module attributes {transform.with_named_sequence} {
  func.func @add_elemwise(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x16xf32>, vector<8x16xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x16xf32>, vector<8x16xf32>
    %2 = arith.addf %0, %1 : vector<8x16xf32>
    %3 = vector.transfer_write %2, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, tensor<8x16xf32>
    return %3 : tensor<8x16xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield 
  }
}

