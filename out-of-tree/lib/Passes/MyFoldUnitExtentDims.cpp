//===- MyFoldUnitExtentDims.cpp - in-tree unit-dim fold 재현 -*- C++ -*-===//
//
// #0007 [linalg pass 해부 cycle] linalg-fold-unit-extent-dims 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgFoldUnitExtentDimsPass               DropUnitDims.cpp:817-835
//       (def: Passes.td:61-71 — Pass<"linalg-fold-unit-extent-dims", "">,
//        anchor "" = op-agnostic. 옵션 1개:
//        useRankReducingSlices / "use-rank-reducing-slices" / bool /
//        default false — "Generate rank-reducing slices instead of
//        reassociative reshapes".
//        dependentDialects = ["linalg::LinalgDialect", "affine::AffineDialect",
//                             "memref::MemRefDialect"])
//       └─ runOnOperation()                      DropUnitDims.cpp:822-834
//            ├─ ControlDropUnitDims options      (:826, Transforms.h:473-489)
//            │    · rankReductionStrategy 기본값 = ReassociativeReshape
//            │      (Transforms.h:476-477)
//            │    · controlFn 기본값 (Transforms.h:479-488) = GenericOp 의
//            │      모든 loop dim / tensor.PadOp 의 모든 source dim 을 허용
//            ├─ useRankReducingSlices == true 이면
//            │    options.rankReductionStrategy = ExtractInsertSlice (:827-830)
//            │    ★ 옵션이 가르는 유일한 분기점 — 이후 populate 선택과
//            │      collapse/expand 의 op 종류가 모두 여기서 갈린다.
//            ├─ populateFoldUnitExtentDimsPatterns(patterns, options)   :831
//            │    선언 Transforms.h:1715-1716 / 정의 DropUnitDims.cpp:798-808
//            │    strategy 로 2-way 분기:
//            │    ┌─ ReassociativeReshape (기본) →
//            │    │  populateFoldUnitExtentDimsViaReshapesPatterns (:762-780)
//            │    │    ├─ DropUnitDims                       (:553-566) ★본체
//            │    │    │    matchAndRewrite(:558-561) = 단순 위임 →
//            │    │    │    linalg::dropUnitDims(rewriter, genericOp, options)
//            │    │    │      (정의 :389-550, 선언 Transforms.h:491-492)
//            │    │    │      1. inversePermutation(concatAffineMaps(maps)) 로
//            │    │    │         iteration dim→operand dim 역사상, static
//            │    │    │         size==1 ∧ controlFn 허용 → unitDims (:395-420)
//            │    │    │      2. unit dim 은 AffineConstantExpr(0) 치환,
//            │    │    │         나머지는 재번호 → newIteratorTypes (:422-440)
//            │    │    │      3. operand 별 dropUnitExtentFromOperandMetadata
//            │    │    │         (:340-387) — 새 map·targetShape·reassociation
//            │    │    │         (hasCollapsibleType(:457-466): identity layout
//            │    │    │          memref / encoding 없는 tensor 만)
//            │    │    │      4. collapseValue(:284-328) — Reshape 전략이면
//            │    │    │         tensor/memref.collapse_shape, Slice 전략이면
//            │    │    │         tensor.extract_slice / memref.subview 의
//            │    │    │         rankReduceIfNeeded
//            │    │    │      5. 새 GenericOp + region inline (:512-526)
//            │    │    │      5a. replaceUnitDimIndexOps(:232-251) —
//            │    │    │          dropped dim 의 linalg.index → const 0,
//            │    │    │          나머지 index 는 번호 시프트
//            │    │    │      6. expandValue(:256-279) — 결과를 원형으로
//            │    │    │         (expand_shape 또는 insert_slice) → replaceOp
//            │    │    ├─ DropPadUnitDims                    (:573-685)
//            │    │    │    tensor.pad 의 size==1 ∧ low/high==0 인 dim 을
//            │    │    │    collapse 하고 pad 후 다시 expand
//            │    │    ├─ RankReducedExtractSliceOp          (:690-720)
//            │    │    │    extract_slice 결과의 unit dim 을 rank-reduced
//            │    │    │    slice + expand_shape 로
//            │    │    ├─ RankReducedInsertSliceOp<InsertSliceOp>
//            │    │    │  RankReducedInsertSliceOp<ParallelInsertSliceOp>
//            │    │    │    (:724-757) — source 를 collapse_shape 후 삽입
//            │    │    └─ 보조: FillOp/CollapseShapeOp/EmptyOp/ExpandShapeOp
//            │    │       canonicalization (:773-776) +
//            │    │       tensor::populateFoldTensorEmptyPatterns (:777) +
//            │    │       memref::populateResolve{Ranked,}ShapedTypeResult
//            │    │       DimsPatterns (:778-779)
//            │    └─ ExtractInsertSlice →
//            │       populateFoldUnitExtentDimsViaSlicesPatterns (:782-796)
//            │         options.rankReductionStrategy 를 강제 재설정(:786-787) 후
//            │         DropUnitDims + DropPadUnitDims 만 (RankReduced*SliceOp
//            │         3종과 CollapseShape/ExpandShape canonicalization 은
//            │         **추가하지 않음**) + FillOp/EmptyOp canonicalization +
//            │         FoldTensorEmpty + memref resolve-dims (:791-795)
//            ├─ populateMoveInitOperandsToInputPattern(patterns)        :832
//            │    선언 Transforms.h:1719 / 정의 DropUnitDims.cpp:810-813
//            │    └─ MoveInitOperandsToInput               (:83-163)
//            │         pure-tensor ∧ all-parallel generic 에서 body 가 읽는
//            │         init(outs) 을 ins 로 옮기고 outs 는 새 tensor.empty 로
//            │         — unit reduction dim 이 모두 fold 되어 elementwise 가
//            │         된 op 의 후처리 (DropUnitDims 와 같은 greedy 안에서 합주)
//            └─ applyPatternsAndFoldGreedily(op, std::move(patterns))   :833
//                 → **greedy driver**, GreedyRewriteConfig 기본값
//
//   주의: 같은 파일 끝의 populateContractionOpRankReducingPatterns
//   (:1067-1097, RankReduceMatmul/RankReduceToUnBatched — named contraction
//   op 의 unit dim 강하) 는 이 pass 의 runOnOperation 에 **포함되지 않는다**.
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// linalg::populateFoldUnitExtentDimsPatterns /
// linalg::populateMoveInitOperandsToInputPattern (둘 다 MLIRLinalgTransforms)
// 를 #include 로 가져와 in-tree runOnOperation() 과 *동일한 절차·동일한
// 순서* 로 호출한다. 옵션 use-rank-reducing-slices 도 in-tree 와 같은 이름·
// 같은 기본값(false)으로 노출한다.
//
// 핵심 학습 포인트:
//   - unit dim 판정의 정수는 **indexing map 의 역사상**:
//     inversePermutation(concatAffineMaps(maps)) 로 "iteration dim d 가
//     어느 operand 의 어느 자리에 닿는가" 를 얻고, 그 자리 static size 가 1
//     이면 d 는 one-trip loop → AffineConstantExpr(0) 으로 대수적 치환.
//     (affine 의 trip-count 분석이 아니라 map 대수로 똑같은 결론을 얻는다.)
//   - RankReductionStrategy 는 *동일한 수학적 변환* 의 표현 선택:
//     ReassociativeReshape = collapse_shape/expand_shape (메타데이터 변환),
//     ExtractInsertSlice = extract_slice/insert_slice (복사 의미론) —
//     bufferization 친화성이 다르다.
//   - ControlDropUnitDims.controlFn 훅: 어떤 dim 을 떨어뜨릴지 호출자가
//     선별 가능 (기본 = 전부). downstream 은 layout 제약 있는 dim 만 보존.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h" // dependentDialects (Passes.td:69)
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1715,:1719)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

struct MyFoldUnitExtentDimsPass
    : public PassWrapper<MyFoldUnitExtentDimsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFoldUnitExtentDimsPass)

  MyFoldUnitExtentDimsPass() = default;
  MyFoldUnitExtentDimsPass(const MyFoldUnitExtentDimsPass &pass)
      : PassWrapper(pass) {}

  // in-tree Passes.td:64-67 과 동일한 이름·타입·기본값의 옵션.
  // (tablegen 이 PassBase 에 생성해 주는 Option<> 멤버를 PassWrapper 에서는
  //  직접 선언한다. clonePass 가 copyOptionValuesFrom 으로 값을 복사하므로
  //  copy ctor 만 base-forwarding 으로 두면 된다.)
  Option<bool> useRankReducingSlices{
      *this, "use-rank-reducing-slices",
      llvm::cl::desc(
          "Generate rank-reducing slices instead of reassociative reshapes"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "my-fold-unit-extent-dims"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-fold-unit-extent-dims: remove "
           "unit-extent dimensions from linalg ops on tensors via "
           "populateFoldUnitExtentDimsPatterns (DropUnitDims/DropPadUnitDims/"
           "RankReduced*SliceOp) + populateMoveInitOperandsToInputPattern + "
           "greedy driver. [#0007 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:68-70 — dependentDialects = linalg, affine, memref.
  // (패턴이 만드는 tensor.collapse_shape/expand_shape/extract_slice 등의
  //  TensorDialect 는 LinalgDialect 의 dependent dialect 로 따라 로드되고,
  //  replaceUnitDimIndexOps 의 arith.constant 도 마찬가지.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  // in-tree DropUnitDims.cpp:822-834 와 한 줄 한 줄 동일한 절차.
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    linalg::ControlDropUnitDims options;
    if (useRankReducingSlices) {
      options.rankReductionStrategy = linalg::ControlDropUnitDims::
          RankReductionStrategy::ExtractInsertSlice;
    }
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    linalg::populateMoveInitOperandsToInputPattern(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyFoldUnitExtentDimsPass() {
  return std::make_unique<MyFoldUnitExtentDimsPass>();
}

} // namespace linalgtransform
