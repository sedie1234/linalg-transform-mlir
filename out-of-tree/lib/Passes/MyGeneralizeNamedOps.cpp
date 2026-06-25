//===- MyGeneralizeNamedOps.cpp - in-tree generalization pass 재현 -*- C++ -*-===//
//
// #0001 [linalg pass 해부 cycle] linalg-generalize-named-ops 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/Generalization.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgGeneralizeNamedOpsPass            Generalization.cpp:79-85
//       └─ runOnOperation()                   Generalization.cpp:89-93
//            ├─ populateLinalgNamedOpsGeneralizationPatterns(patterns)
//            │                                Generalization.cpp:95-98
//            │    └─ patterns.add<LinalgGeneralizationPattern>(ctx)
//            │                                Transforms.h:1408-1421
//            │         └─ matchAndRewrite → generalizeNamedOp(rewriter, op)
//            │                                Generalization.cpp:53-75
//            │              ├─ generalizeNamedOpPrecondition  :38-51
//            │              │    (GenericOp/MapOp 이면 bail, region != 1 bail)
//            │              ├─ DPS inputs/inits + indexingMaps + iterators 추출
//            │              ├─ rewriter.create<GenericOp>(...)  :69-70
//            │              ├─ rewriter.inlineRegionBefore(...) :71-72
//            │              └─ rewriter.replaceOp(...)          :73
//            └─ applyPatternsAndFoldGreedily(getOperation(), patterns)
//                                             Generalization.cpp:92  (greedy driver)
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateLinalgNamedOpsGeneralizationPatterns() (선언 Transforms.h:1588,
// 정의 Generalization.cpp:95, lib MLIRLinalgTransforms) 를 #include 로 가져와
// in-tree runOnOperation() 과 *동일한 절차* 로 호출한다.
//
// 핵심 학습 포인트:
//   - 이 pass 는 "가장 단순한 pattern pass" 표본이다: 옵션 0개, pattern 1개,
//     driver 는 greedy 한 줄.  pattern pass 해부의 기준선.
//   - LinalgGeneralizationPattern 은 OpInterfaceRewritePattern<LinalgOp> —
//     특정 named op 가 아니라 LinalgOp *인터페이스* 전체에 match 하므로
//     matmul/add/transpose/fill/conv 등 모든 named op 가 한 pattern 으로 처리.
//   - 변환의 본질은 "정보 보존 + 표현만 하향": named op 가 이미 들고 있는
//     indexing_maps/iterator_types/payload region 을 *그대로* generic 으로
//     옮길 뿐이다 (region 은 inlineRegionBefore 로 이동, 재생성 아님).
//   - in-tree pass def (Passes.td:92-95) 는 anchor 없는 Pass<"..."> 이므로
//     여기서도 OperationPass<> (any-op anchor) 를 쓴다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1588)
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily

using namespace mlir;

namespace {

struct MyGeneralizeNamedOpsPass
    : public PassWrapper<MyGeneralizeNamedOpsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyGeneralizeNamedOpsPass)

  StringRef getArgument() const final { return "my-generalize-named-ops"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-generalize-named-ops: convert linalg "
           "named ops into linalg.generic via "
           "populateLinalgNamedOpsGeneralizationPatterns + greedy driver. "
           "[#0001 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:94 — dependentDialects = ["linalg::LinalgDialect"]
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  // in-tree Generalization.cpp:89-93 과 한 줄 한 줄 동일한 절차.
  // (옵션 없음 — in-tree pass 도 옵션이 없다.)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyGeneralizeNamedOpsPass() {
  return std::make_unique<MyGeneralizeNamedOpsPass>();
}

} // namespace linalgtransform
