// #0009 입력 3 — negative: tensor semantics 의 linalg.matmul.
//
// LinalgRewritePattern::matchAndRewrite 의 가드 (Loops.cpp:266-269)
//   !linalgOp.hasPureBufferSemantics() →
//   notifyMatchFailure "expected linalg op with buffer semantics"
// 에 걸려 **발화하지 않는다**. 세 모드 모두 IR 불변이어야 한다.
// (pre-condition: 이 pass 앞에 bufferization 이 와야 한다 — Passes.td:36-41.)
func.func @matmul_tensor(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>,
                         %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>)
                     outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
