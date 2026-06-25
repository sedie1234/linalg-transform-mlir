//===- MyFuseElementwiseOps.cpp - in-tree elementwise fusion 재현 -*- C++ -*-===//
//
// #0006 [linalg pass 해부 cycle] linalg-fuse-elementwise-ops 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgElementwiseOpFusionPass            ElementwiseOpFusion.cpp:2128-2164
//       (def: Passes.td:73-78 — Pass<"linalg-fuse-elementwise-ops">,
//        anchor 없음 = op-agnostic, 옵션 0개,
//        dependentDialects = ["affine::AffineDialect", "linalg::LinalgDialect",
//                             "memref::MemRefDialect"])
//       └─ runOnOperation()                    ElementwiseOpFusion.cpp:2133-2163
//            ├─ defaultControlFn (:2139-2142) — ControlFusionFn
//            │    = [](OpOperand *fusedOperand) {
//            │        producer = fusedOperand->get().getDefiningOp();
//            │        return producer && producer->hasOneUse(); }
//            │    → "producer 결과가 단일 사용일 때만 fuse" (복제 비용 회피).
//            │    모든 populate 에 같은 controlFn 이 들어간다.
//            ├─ populateElementwiseOpsFusionPatterns(patterns, ctrl)    :2145
//            │    선언 Transforms.h:1656 / 정의 ElementwiseOpFusion.cpp:2097-2106
//            │    ├─ FuseElementwiseOps                       (:417-456) ★본체
//            │    │    matchAndRewrite(:424): consumer GenericOp 의 각 operand 에
//            │    │      ├─ areElementwiseOpsFusable(&opOperand)   (:93-167)
//            │    │      │    (선언 Transforms.h:452 — producer/consumer 모두
//            │    │      │     GenericOp + producer pure-tensor + producer
//            │    │      │     all-parallel + DPS input + consumerIndexMap
//            │    │      │     결과수==producer 루프수 + producer result map 이
//            │    │      │     permutation + reduction 시 dim coverage 검사)
//            │    │      ├─ controlFn(&opOperand)                   (:430)
//            │    │      └─ fuseElementwiseOps(rewriter, &opOperand) (:292-413)
//            │    │           (선언 Transforms.h:503-504)
//            │    │           ├─ getPreservedProducerResults        (:76-90)
//            │    │           ├─ operand/indexing-map 병합 — producer 입력 map 은
//            │    │           │   getIndexingMapOfProducerOperandsInCoordinates
//            │    │           │   OfFusedOp(:44-71): argMap ∘ inv(producerResultMap)
//            │    │           │   ∘ consumerArgMap 의 AffineMap 합성
//            │    │           ├─ rewriter.create<GenericOp>(병합 결과) (:373)
//            │    │           └─ generateFusedElementwiseOpRegion    (:171-290)
//            │    │               (양쪽 payload block 을 한 block 으로 splice,
//            │    │                linalg.index 는 consumerToProducerLoopsMap 으로
//            │    │                remap)
//            │    ├─ FoldFillWithGenericOp                  (:2047-2074)
//            │    │    linalg.fill 결과를 읽는 input 을 fill 스칼라로 치환
//            │    ├─ FoldScalarOrSplatConstant              (:1893-1993)
//            │    │    splat/scalar arith.constant input 을 body 안 스칼라
//            │    │    상수로 끌어들이고 operand 에서 제거
//            │    ├─ RemoveOutsDependency                   (:2006-2044)
//            │    │    payload 가 안 읽는 outs 를 tensor.empty 로 강제
//            │    │    (init 의존 제거 → fusion 기회 확대)
//            │    └─ populateEraseUnusedOperandsAndResultsPatterns
//            │         (선언 Transforms.h:1671 / 정의
//            │          EraseUnusedOperandsAndResults.cpp:421 — dead 피연산자/
//            │          결과 제거 청소 패턴)
//            ├─ populateFoldReshapeOpsByExpansionPatterns(patterns, ctrl) :2146
//            │    선언 Transforms.h:1692 / 정의 ElementwiseOpFusion.cpp:2077-2086
//            │    ├─ FoldReshapeWithGenericOpByExpansion     (:1024-1088)
//            │    │    generic 결과를 먹는 tensor.expand_shape 를 generic 안으로
//            │    ├─ FoldPadWithProducerReshapeOpByExpansion (:959-1020)
//            │    │    collapse_shape → tensor.pad 를 pad → collapse_shape 로
//            │    └─ FoldWithProducerReshapeOpByExpansion    (:922-957)
//            │         input 의 tensor.collapse_shape 를 generic 안으로 —
//            │         셋 다 핵심은 fuseWithReshapeByExpansion(:775-920)
//            │         (ExpansionInfo(:544) 로 루프 차원을 확장해 reshape 흡수)
//            ├─ canonicalization 패턴 (:2149-2154):
//            │    affine::AffineApplyOp / GenericOp / tensor::ExpandShapeOp /
//            │    tensor::CollapseShapeOp::getCanonicalizationPatterns +
//            │    LinalgDialect->getCanonicalizationPatterns
//            ├─ populateConstantFoldLinalgOperations(patterns, ctrl)    :2157
//            │    선언 Transforms.h:1701 / 정의 ConstantFold.cpp:306
//            │    (모든 input 이 상수인 generic 을 컴파일타임 평가)
//            └─ applyPatternsAndFoldGreedily(op, patterns, grc)   :2160-2162
//                 → **greedy driver**, GreedyRewriteConfig
//                   useTopDownTraversal = true (컴파일 시간 사유 주석 :2159)
//
//   주의: 같은 파일의 collapse 계열 (populateFoldReshapeOpsByCollapsing
//   Patterns :2088-2095, populateCollapseDimensions :2108-2114) 은 이 pass 의
//   runOnOperation 에 **포함되지 않는다** — expansion 방향만 쓴다.
//   in-tree 주석 (:2122-2127): 이 pass 는 테스트용이며 cost function 에 따라
//   패턴 효과가 갈리므로 deprecate 예정 — 실전은 populate 함수를 직접 조합.
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateElementwiseOpsFusionPatterns / populateFoldReshapeOpsByExpansion
// Patterns / populateConstantFoldLinalgOperations (모두 MLIRLinalgTransforms)
// 와 각 dialect 의 getCanonicalizationPatterns 를 #include 로 가져와
// in-tree runOnOperation() 과 *동일한 절차·동일한 순서* 로 호출한다.
// (옵션 없음 — in-tree pass 도 옵션이 없다. controlFn 도 in-tree 의
//  defaultControlFn 을 줄 단위로 동일하게 둔다.)
//
// 핵심 학습 포인트:
//   - producer-consumer fusion 의 정수는 **AffineMap 합성**:
//     producer arg map ∘ inv(producer result map) ∘ consumer arg map
//     으로 "consumer 루프 좌표 → producer arg 텐서 좌표" 를 만든다.
//     (affine #0023 의 ad-hoc fusion 과 달리 의존성 분석이 아니라
//      indexing map 의 대수적 합성으로 합법성·결과를 동시에 얻는다.)
//   - ControlFusionFn 훅: 합법성(areElementwiseOpsFusable)과 별개로
//     "할지 말지" 의 비용 판단을 호출자에게 위임하는 콜백. in-tree pass 는
//     hasOneUse 휴리스틱, downstream 컴파일러는 자체 cost model 을 꽂는다.
//   - 이 pass 는 fusion 본체 1개가 아니라 패턴 11+개의 합주:
//     fusion(1) + fold(fill/splat/outs/erase 4) + reshape 전파(3) +
//     canonicalization(5 그룹) + constant fold(1) 가 greedy 고정점까지 돈다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h" // AffineApplyOp + AffineDialect
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1656,:1692,:1701)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // ExpandShapeOp/CollapseShapeOp canonicalization
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct MyFuseElementwiseOpsPass
    : public PassWrapper<MyFuseElementwiseOpsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFuseElementwiseOpsPass)

  StringRef getArgument() const final { return "my-fuse-elementwise-ops"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-fuse-elementwise-ops: fuse elementwise "
           "linalg.generic producer-consumer pairs (plus reshape-by-expansion "
           "propagation, fill/splat/outs folds, canonicalizations, constant "
           "folding) via populateElementwiseOpsFusionPatterns + "
           "populateFoldReshapeOpsByExpansionPatterns + greedy driver "
           "(top-down). [#0006 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:75-77 — dependentDialects = affine, linalg, memref.
  // (패턴이 만드는 tensor.empty/expand_shape/collapse_shape 의 TensorDialect
  //  는 LinalgDialect 의 dependent dialect 로 따라 로드된다.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  // in-tree ElementwiseOpFusion.cpp:2133-2163 과 한 줄 한 줄 동일한 절차.
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    // Add folding with reshape by expansion patterns.
    // (in-tree :2139-2142 의 defaultControlFn 과 동일 — producer 단일 사용.)
    linalg::ControlFusionFn defaultControlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer && producer->hasOneUse();
    };

    // Add elementwise op fusion patterns. (in-tree :2145-2146)
    linalg::populateElementwiseOpsFusionPatterns(patterns, defaultControlFn);
    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns,
                                                      defaultControlFn);

    // General canonicalization patterns. (in-tree :2149-2154)
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    context->getLoadedDialect<linalg::LinalgDialect>()
        ->getCanonicalizationPatterns(patterns);

    // Add constant folding patterns. (in-tree :2157)
    linalg::populateConstantFoldLinalgOperations(patterns, defaultControlFn);

    // Use TopDownTraversal for compile time reasons. (in-tree :2159-2162)
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyFuseElementwiseOpsPass() {
  return std::make_unique<MyFuseElementwiseOpsPass>();
}

} // namespace linalgtransform
