// B — match matmul, then get_producer_of_operand %mm[2] to grab the op that
//     produces matmul's outs operand (#2 = %filled, the linalg.fill).
//     Proof: generalize ONLY that producer handle. If the handle truly points
//     at the fill, the fill becomes a linalg.generic while the matmul stays
//     a named linalg.matmul.
module attributes {transform.with_named_sequence} {
  func.func @mm(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
                       outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // operand #2 of linalg.matmul is the outs tensor, produced by linalg.fill.
    %producer = transform.get_producer_of_operand %mm[2]
      : (!transform.any_op) -> !transform.any_op
    transform.print %producer {name = "producer of matmul outs (expect linalg.fill)"} : !transform.any_op
    %g = transform.structured.generalize %producer : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
