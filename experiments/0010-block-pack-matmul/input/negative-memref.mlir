// #0010 negative — buffer semantics (memref) matmul.
// blockPackMatmul (BlockPackMatmul.cpp:141-142) 의
//   if (linalgOp.hasPureBufferSemantics())
//     return rewriter.notifyMatchFailure(linalgOp, "require tensor semantics");
// 에 걸려 block-factors 를 줘도 절대 발화하지 않는다 → no-op 검증용.
func.func @memref_matmul(%A: memref<64x64xf32>, %B: memref<64x64xf32>,
                         %C: memref<64x64xf32>) {
  linalg.matmul ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
                outs(%C : memref<64x64xf32>)
  return
}
