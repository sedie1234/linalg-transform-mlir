// B — convert_to_loops on a memref-semantic matmul.
// convert_to_loops uses the TilingInterface generateScalarImplementation hook,
// which requires buffer (memref) operands. It lowers the matmul to a 3-level
// scf.for nest of memref.load / arith.mulf / arith.addf / memref.store.
module attributes {transform.with_named_sequence} {
  func.func @matmul_buf(%A: memref<128x256xf32>, %B: memref<256x64xf32>, %C: memref<128x64xf32>) {
    linalg.matmul ins(%A, %B : memref<128x256xf32>, memref<256x64xf32>) outs(%C : memref<128x64xf32>)
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.convert_to_loops %mm : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
