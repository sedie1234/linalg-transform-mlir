// C — full pipeline on the tensor matmul: bufferize_to_allocation +
// one_shot_bufferize + convert_to_loops.
//   1. bufferize_to_allocation: explicit memref.alloc for the accumulator.
//   2. one_shot_bufferize on the module: lowers remaining tensor ops to memref
//      (to_memref on the operands, the matmul becomes memref-semantic).
//   3. convert_to_loops: the now buffer-semantic matmul lowers to the scf.for
//      nest. Result keeps the alloc/copy/dealloc + to_tensor at the boundary.
// arg0 is consumed because one_shot_bufferize rewrites the module in place.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %C: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>) outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %buf, %new = transform.structured.bufferize_to_allocation %mm {memory_space = 0, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %bufmod = transform.bufferization.one_shot_bufferize %arg0 : (!transform.any_op) -> !transform.any_op
    %mm2 = transform.structured.match ops{["linalg.matmul"]} in %bufmod : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.convert_to_loops %mm2 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
