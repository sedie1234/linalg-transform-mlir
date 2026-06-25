//===- MyLinalgToLoopsTrio.cpp - in-tree linalg→loops 3종 재현 -*- C++ -*-===//
//
// #0009 [linalg pass 해부 cycle] convert-linalg-to-loops /
// convert-linalg-to-affine-loops / convert-linalg-to-parallel-loops 재현.
// 세 in-tree pass 는 **한 template 기계** (linalgOpToLoopsImpl<LoopTy>) 의
// 3 instantiation 이므로 본인 pass 는 1개로 통합하고 `mode` 옵션
// (scf | affine | parallel, 기본 scf) 으로 LoopTy 를 고른다.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/Loops.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LowerToLoops               Loops.cpp:339-348 (scf.for)
//     LowerToAffineLoops         Loops.cpp:327-337 (affine.for)
//     LowerToParallelLoops       Loops.cpp:350-357 (scf.parallel)
//       (def: Passes.td:33/26/49 — Pass<"convert-linalg-to-*loops">,
//        anchor 없음 = op-agnostic. **옵션 0개.** dependentDialects 는
//        td(:29-30,:42-46,:52-58) + C++ getDependentDialects override
//        (:331-333 memref / :342-344 memref+scf) 의 합집합)
//       └─ runOnOperation() → lowerLinalgToLoopsImpl<LoopType>   :314-325
//            ├─ patterns.add<LinalgRewritePattern<LoopType>>     :318
//            │    (file-local class :257-275, MatchAnyOpTypeTag —
//            │     모든 op 에 매치 시도, benefit 1)
//            │    matchAndRewrite                                :263-274
//            │      · dyn_cast<LinalgOp> 실패 ∨ !hasPureBufferSemantics()
//            │        → notifyMatchFailure "expected linalg op with buffer
//            │        semantics" (:266-269) ★ tensor semantics 는 발화 X
//            │      · linalgOpToLoopsImpl<LoopType>(rewriter, linalgOp) :270
//            │        (:208-254 — 본체. 아래 ①~④)
//            │      · 성공 시 rewriter.eraseOp(op)               :272
//            ├─ memref::DimOp / tensor::DimOp /
//            │  affine::AffineApplyOp getCanonicalizationPatterns :319-321
//            ├─ patterns.add<FoldAffineOp>                       :322
//            │    (file-local :287-312 — trivial affine.apply 를 상수/
//            │     유일 operand 로 접는 local fold. 아래 verbatim 복제)
//            └─ applyPatternsAndFoldGreedily                     :324
//                 → **greedy driver**, GreedyRewriteConfig 기본값
//
//   linalgOpToLoopsImpl<LoopTy> (:208-254) 의 절차:
//     ① LoadOpTy/StoreOpTy 선택 (:211-216) — LoopTy==affine::AffineForOp
//        이면 affine.load/store, 아니면 memref.load/store
//     ② linalgOp.createLoopRanges (LinalgInterfaces.cpp:994-1009) —
//        getLoopsToShapesMap() 의 AffineDimExpr 결과 자리마다
//        Range{0, viewSizes[idx], 1} (loop dim ↔ operand shape 대응)
//     ③ GenerateLoopNest<LoopTy>::doit (선언 Utils.h:356-365, 정의
//        Utils.cpp:313/356/523) — scf.for 는 scf::buildLoopNest,
//        affine.for 는 affine::buildAffineLoopNest(step 상수 필수),
//        scf.parallel 은 generateParallelLoopNest(Utils.cpp:408-520)
//        재귀: 연속 parallel iterator 묶음 → 1개 scf.parallel,
//        reduction iterator → scf.for 로 분리 ★iterator_types 가
//        loop 종류를 가르는 유일한 지점
//        innermost body 에서 emitScalarImplementation (:127-175):
//        · 입력/출력 operand 마다 makeCanonicalAffineApplies (:39-56)
//          — indexing_map 의 result expr 1개씩 AffineApplyOp 생성
//          (affine::canonicalizeMapAndOperands 후) → load index
//        · LoadOpTy 로 load, region 을 IRMapping 으로 clone-inline,
//          yield operand 를 StoreOpTy 로 store
//          (inlineRegionAndEmitStore :58-77)
//     ④ replaceIndexOpsByInductionVariables (:179-206) —
//        linalg.index → 대응 loop iv (scf.parallel 은 iv 여러 개)
//
// **이 pass 3종에는 populate* export 가 없다** — pattern 도 fold 도
// file-local. 대신 in-tree 는 알고리즘 본체를 3개 함수로 export 한다
// (Transforms.h:769/773/777, 정의 Loops.cpp:362-378 — 셋 다
// linalgOpToLoopsImpl<LoopTy> 호출 한 줄):
//   FailureOr<LinalgLoops> linalgOpToLoops(RewriterBase &, LinalgOp);
//   FailureOr<LinalgLoops> linalgOpToParallelLoops(RewriterBase &, LinalgOp);
//   FailureOr<LinalgLoops> linalgOpToAffineLoops(RewriterBase &, LinalgOp);
// 본 파일은 알고리즘을 재구현하지 않는다 — 위 export 함수를 가져와
// in-tree LinalgRewritePattern 과 같은 골격의 thin pattern 에서 호출한다
// (in-tree pattern 의 :270 직접 호출과 동일 코드 경로). file-local
// FoldAffineOp (25줄 fold helper) 만 Loops.cpp:287-312 그대로 복제.
//
// 핵심 학습 포인트:
//   - linalg 의 "계산 의미" 3요소가 그대로 loop 가 된다:
//     iterator_types → loop 종류(scf.parallel 묶음/scf.for),
//     indexing_maps → load/store index (affine.apply 로 전개),
//     region → innermost body (clone-inline).
//   - affine 모드가 affine.load/store 를 쓰는 이유: affine dialect 의
//     구조 제약(인덱스가 affine map 의 결과여야 함)을 op 단위로 보장 —
//     이후 affine 분석(dependence 등)이 가능한 IR 이 된다.
//   - FoldAffineOp + AffineApplyOp canonicalization 이 같은 greedy 안에
//     있어서, identity map 의 affine.apply(d0)->d0 은 IR 에 남지 않고
//     iv 가 직접 load/store index 로 들어간다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h" // AffineApplyOp canonicalization
#include "mlir/Dialect/Arith/IR/Arith.h"      // FoldAffineOp 의 ConstantIndexOp
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // linalgOpTo*Loops (:769-778)
#include "mlir/Dialect/MemRef/IR/MemRef.h"             // memref::DimOp canon.
#include "mlir/Dialect/SCF/IR/SCF.h"                   // dependent dialect
#include "mlir/Dialect/Tensor/IR/Tensor.h"             // tensor::DimOp canon.
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

/// in-tree 가 export 하는 알고리즘 본체 3개 (linalgOpToLoops /
/// linalgOpToAffineLoops / linalgOpToParallelLoops) 의 공통 시그니처.
using LoopLoweringFn = FailureOr<linalg::LinalgLoops> (*)(RewriterBase &,
                                                          linalg::LinalgOp);

/// in-tree LinalgRewritePattern<LoopType> (Loops.cpp:257-275) 와 같은 골격.
/// 차이는 단 하나 — in-tree 는 file-local 인 linalgOpToLoopsImpl<LoopType>
/// 을 직접 호출(:270)하고, 본 pattern 은 그 함수의 export wrapper
/// (Loops.cpp:362-378) 를 함수 포인터로 받아 호출한다. 코드 경로 동일.
class MyLinalgRewritePattern : public RewritePattern {
public:
  MyLinalgRewritePattern(MLIRContext *context, LoopLoweringFn loweringFn)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        loweringFn(loweringFn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // in-tree :265-269 와 동일한 가드 — buffer semantics 인 LinalgOp 만.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          op, "expected linalg op with buffer semantics");
    }
    if (failed(loweringFn(rewriter, linalgOp)))
      return failure();
    rewriter.eraseOp(op); // in-tree :272
    return success();
  }

private:
  LoopLoweringFn loweringFn;
};

/// Local folding pattern for AffineApplyOp that we can apply greedily.
/// This replaces AffineApplyOp by the proper value in cases where the
/// associated map is trivial.
/// A trivial map here is defined as a map with a single result and either:
///   1. Zero operand + returns a single AffineConstantExpr
///   2. One operand + returns a single AffineDimExpr
///   3. One operand + returns a single AffineSymbolExpr
//
/// In the first case, the AffineApplyOp is replaced by a new constant. In the
/// other cases, it is replaced by its unique operand.
///
/// (in-tree Loops.cpp:287-312 verbatim — file-local 이라 import 불가.)
struct FoldAffineOp : public RewritePattern {
  FoldAffineOp(MLIRContext *context)
      : RewritePattern(affine::AffineApplyOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto affineApplyOp = cast<affine::AffineApplyOp>(op);
    auto map = affineApplyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumInputs() > 1)
      return failure();

    AffineExpr expr = map.getResult(0);
    if (map.getNumInputs() == 0) {
      if (auto val = dyn_cast<AffineConstantExpr>(expr)) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, val.getValue());
        return success();
      }
      return failure();
    }
    if (dyn_cast<AffineDimExpr>(expr) || dyn_cast<AffineSymbolExpr>(expr)) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    return failure();
  }
};

struct MyLinalgToLoopsTrioPass
    : public PassWrapper<MyLinalgToLoopsTrioPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyLinalgToLoopsTrioPass)

  MyLinalgToLoopsTrioPass() = default;
  MyLinalgToLoopsTrioPass(const MyLinalgToLoopsTrioPass &pass)
      : PassWrapper(pass) {}

  // in-tree 3 pass 는 옵션이 없다 (Passes.td:26-59). 본 pass 는 3종을 1개로
  // 통합했으므로 LoopTy 선택자만 옵션으로 노출한다:
  //   mode=scf      ↔ convert-linalg-to-loops          (LoopTy=scf::ForOp)
  //   mode=affine   ↔ convert-linalg-to-affine-loops   (LoopTy=affine::AffineForOp)
  //   mode=parallel ↔ convert-linalg-to-parallel-loops (LoopTy=scf::ParallelOp)
  Option<std::string> mode{
      *this, "mode",
      llvm::cl::desc("Loop type to lower to: scf (scf.for, default) | affine "
                     "(affine.for) | parallel (scf.parallel for parallel "
                     "iterators, scf.for for reductions)"),
      llvm::cl::init("scf")};

  StringRef getArgument() const final { return "my-linalg-to-loops-trio"; }

  StringRef getDescription() const final {
    return "Replicate in-tree convert-linalg-to-{loops,affine-loops,"
           "parallel-loops}: lower linalg ops with buffer semantics to "
           "explicit loop nests via linalgOpTo{Loops,AffineLoops,"
           "ParallelLoops} + DimOp/AffineApplyOp canonicalization + "
           "FoldAffineOp + greedy driver. [#0009 linalg-transform-mlir 학습용]";
  }

  // in-tree 3 pass 의 dependentDialects 합집합:
  //   td(:29-30) affine,linalg,memref / td(:42-46) linalg,scf,affine /
  //   td(:52-58) affine,linalg,memref,scf
  //   + C++ override(:331-333,:342-344) memref,scf.
  // (arith 는 affine/memref dialect 의 dependent 로 따라 로드된다.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  // in-tree lowerLinalgToLoopsImpl<LoopType> (Loops.cpp:314-325) 와
  // 한 줄 한 줄 동일한 절차 — pattern 의 알고리즘 본체만 mode 로 선택.
  void runOnOperation() override {
    Operation *op = getOperation();

    LoopLoweringFn loweringFn;
    if (mode == "scf") {
      loweringFn = linalg::linalgOpToLoops; // Loops.cpp:368-371
    } else if (mode == "affine") {
      loweringFn = linalg::linalgOpToAffineLoops; // Loops.cpp:362-365
    } else if (mode == "parallel") {
      loweringFn = linalg::linalgOpToParallelLoops; // Loops.cpp:374-378
    } else {
      op->emitError("my-linalg-to-loops-trio: unknown mode \"")
          << mode << "\" (expected scf | affine | parallel)";
      return signalPassFailure();
    }

    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<MyLinalgRewritePattern>(context, loweringFn); // :318
    memref::DimOp::getCanonicalizationPatterns(patterns, context);   // :319
    tensor::DimOp::getCanonicalizationPatterns(patterns, context);   // :320
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context); // :321
    patterns.add<FoldAffineOp>(context); // :322
    // Just apply the patterns greedily. (in-tree :324)
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyLinalgToLoopsTrioPass() {
  return std::make_unique<MyLinalgToLoopsTrioPass>();
}

} // namespace linalgtransform
