// A — bufferize_to_allocation on the tensor matmul's destination operand.
// Allocates an explicit memref for the accumulator (outs), materializes the
// init tensor into it, and emits a matching dealloc. The matmul itself stays
// in tensor form (bufferize_destination_only). No loops yet.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %C: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>) outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %buf, %new = transform.structured.bufferize_to_allocation %mm {memory_space = 0, bufferize_destination_only, emit_dealloc} : !transform.any_op
    transform.yield
  }
}
