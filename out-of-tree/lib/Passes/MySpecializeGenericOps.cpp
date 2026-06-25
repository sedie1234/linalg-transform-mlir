//===- MySpecializeGenericOps.cpp - in-tree specialization pass 재현 -*- C++ -*-===//
//
// #0002 [linalg pass 해부 cycle] linalg-specialize-generic-ops 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/Specialize.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgSpecializeGenericOpsPass           Specialize.cpp:312-319
//       └─ runOnOperation()                    Specialize.cpp:322-328
//            ├─ populateLinalgGenericOpsSpecializationPatterns(patterns)
//            │                                 Specialize.cpp:330-333
//            │    └─ patterns.add<LinalgSpecializationPattern>(ctx)
//            │                                 Transforms.h:1425-1437
//            │         └─ matchAndRewrite → specializeGenericOp(rewriter, op)
//            │              선언 Transforms.h:698 / 정의 Specialize.cpp:262-309
//            │              ── idiom 인식 분기 (이 순서대로 검사) ──
//            │              ├─ isaCopyOpInterface       :264  → linalg.copy
//            │              │    (LinalgInterfaces.cpp:56-71)
//            │              ├─ isaFillOpInterface       :270  → linalg.fill
//            │              │    (LinalgInterfaces.cpp:76-100)
//            │              ├─ isaElemwiseSingleUnaryOpInterface :276
//            │              │    (LinalgInterfaces.cpp:142-151) — math.exp 만
//            │              │    → linalg.exp  (REPLACE_UNARY_OP :39-42)
//            │              ├─ isaElemwiseSingleBinaryOpInterface :284
//            │              │    (LinalgInterfaces.cpp:153-164)
//            │              │    ├─ areBinOpsSwapped :58-69 (operand 순서 판정)
//            │              │    └─ arith.{addf,subf,mulf,divf}
//            │              │       → linalg.{add,sub,mul,div}
//            │              │         (REPLACE_BINARY_OP :32-37, swap 반영)
//            │              └─ isaContractionOpInterface :305
//            │                   (LinalgInterfaces.cpp:453-460)
//            │                   └─ specializeLinalgContractions :148-255
//            │                        ├─ 2-in/1-out + projectedPermutation :150-157
//            │                        ├─ inferContractionDims → m,n,k 각 1개 :178-183
//            │                        ├─ isContractionBody — mulf+addf 류만 :185-193
//            │                        ├─ rank/batch-identity 검사 :196-221
//            │                        ├─ matchOperandMap(A/B/C) :223-228
//            │                        │    (Match/Transposed/Mismatch, :111-133)
//            │                        └─ replaceWithMatmulVariant<T> :139-145
//            │                             → linalg.{batch_}matmul{_transpose_a,_b}
//            └─ applyPatternsAndFoldGreedily(getOperation(), patterns)
//                                              Specialize.cpp:326  (greedy driver)
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateLinalgGenericOpsSpecializationPatterns() (선언 Transforms.h:1596,
// 정의 Specialize.cpp:330, lib MLIRLinalgTransforms) 를 #include 로 가져와
// in-tree runOnOperation() 과 *동일한 절차* 로 호출한다.
//
// 핵심 학습 포인트:
//   - #0001 generalize 의 정확한 역방향. generalize 는 "정보 노출" (named 가
//     이미 가진 maps/iterators/region 을 펼침) 이라 모든 named op 에 무조건
//     발화하지만, specialize 는 "idiom 인식" — generic 의 구조(maps 전부
//     identity 인가, body 가 단일 binary op 인가, contraction dims 가 정확히
//     m/n/k 1개씩인가)를 *판정*해야 하므로 인식 가능한 부분집합에만 발화한다.
//   - 인식 분기 순서는 copy → fill → unary(exp) → binary(add/sub/mul/div)
//     → contraction(matmul 변형 6종). 어디에도 안 걸리면 failure → generic
//     그대로 잔류 (pass 실패 아님).
//   - in-tree pass def (Passes.td:97-100) 는 옵션 0개, anchor 없는 Pass<"...">
//     이므로 여기서도 OperationPass<> (any-op anchor) 를 쓴다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1596)
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily

using namespace mlir;

namespace {

struct MySpecializeGenericOpsPass
    : public PassWrapper<MySpecializeGenericOpsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MySpecializeGenericOpsPass)

  StringRef getArgument() const final { return "my-specialize-generic-ops"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-specialize-generic-ops: convert "
           "linalg.generic back into named ops (copy/fill/exp/add/sub/mul/div/"
           "matmul variants) via "
           "populateLinalgGenericOpsSpecializationPatterns + greedy driver. "
           "[#0002 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:99 — dependentDialects = ["linalg::LinalgDialect"]
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  // in-tree Specialize.cpp:322-328 과 한 줄 한 줄 동일한 절차.
  // (옵션 없음 — in-tree pass 도 옵션이 없다. in-tree 는 greedy 미수렴 시
  //  signalPassFailure() 까지 동일하게 수행한다.)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateLinalgGenericOpsSpecializationPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMySpecializeGenericOpsPass() {
  return std::make_unique<MySpecializeGenericOpsPass>();
}

} // namespace linalgtransform
