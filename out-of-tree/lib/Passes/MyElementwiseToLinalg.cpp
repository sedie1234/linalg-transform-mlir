//===- MyElementwiseToLinalg.cpp - in-tree elementwise→linalg 재현 -*- C++ -*-===//
//
// #0005 [linalg pass 해부 cycle] convert-elementwise-to-linalg 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     ConvertElementwiseToLinalgPass           ElementwiseToLinalg.cpp:122-142
//       (def: Passes.td:14-25 — Pass<"convert-elementwise-to-linalg", "">,
//        anchor 가 빈 문자열 "" = op-agnostic, 옵션 0개,
//        dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"])
//       └─ runOnOperation()                    ElementwiseToLinalg.cpp:128-141
//            ├─ ConversionTarget target(*context)                       :131
//            ├─ populateElementwiseToLinalgConversionPatterns(patterns) :134
//            │       선언 Transforms.h:1642 / 정의 ElementwiseToLinalg.cpp:115-119
//            │    └─ patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>
//            │         (ctx)                                      (pattern 1개)
//            │         RewritePattern(MatchAnyOpTypeTag, benefit=1)     :76-78
//            │         — 특정 op 이 아니라 *모든* op 에 매치 시도
//            │         └─ matchAndRewrite(Operation*, PatternRewriter&) :79-111
//            │              ├─ isElementwiseMappableOpOnRankedTensors(op)
//            │              │  아니면 notifyMatchFailure                :81-83
//            │              ├─ indexingMaps = (numResults+numOperands) 개의
//            │              │  getMultiDimIdentityMap(rank)             :85-88
//            │              ├─ iteratorTypes = rank × parallel          :89-90
//            │              ├─ outputs = getOrCreateOperandsMatchingResultTypes
//            │              │  (rewriter, op)                           :91
//            │              │    (정의 :46-73 — 각 result type t 마다:
//            │              │     같은 type 의 operand 가 있으면 그걸 재사용,
//            │              │     없으면 tensor::EmptyOp 생성. dim 은
//            │              │     tensor::getMixedSizes(b, loc, operands.front())
//            │              │     로 첫 operand 에서 static/dynamic 혼합 추출
//            │              │     → dynamic dim 마다 tensor.dim 이 IR 에 추가됨)
//            │              └─ rewriter.replaceOpWithNewOp<linalg::GenericOp>
//            │                 (..., bodyBuilder)                       :92-109
//            │                   bodyBuilder 가 원래 op 을 *scalar 버전으로 재생성*:
//            │                   builder.create(loc, op->getName().getIdentifier(),
//            │                     regionArgs.take_front(numOperands),
//            │                     resultTypes(elemental), op->getAttrs()) :104-107
//            │                   + linalg::YieldOp                       :108
//            │                   (output block args 는 사용 안 함 — pure init)
//            ├─ target.markUnknownOpDynamicallyLegal([](op) {
//            │       return !isElementwiseMappableOpOnRankedTensors(op);
//            │    })                                                    :135-137
//            └─ applyPartialConversion(func, target, std::move(patterns)) :139
//                 → **dialect-conversion driver** (greedy 아님!)
//                 실패 시 signalPassFailure()                           :139-140
//
//   legality 술어 isElementwiseMappableOpOnRankedTensors
//                                            ElementwiseToLinalg.cpp:24-31:
//     OpTrait::hasElementwiseMappableTraits(op)   (선언 OpDefinition.h:1502,
//       정의 Operation.cpp:1393-1396 — Elementwise && Scalarizable &&
//       Vectorizable && Tensorizable 4개 trait 모두 보유)
//     && 모든 operand type 이 RankedTensorType (llvm::all_of + IsaPred).
//     같은 술어가 (a) pattern 의 match 전제(:81) 와 (b) ConversionTarget
//     의 legality(:135, 부정형) 양쪽에 쓰인다 — partial conversion 에서
//     "illegal = 변환 대상" 과 "pattern 적용 가능" 이 정확히 일치.
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateElementwiseToLinalgConversionPatterns() (선언 Transforms.h:1642,
// lib MLIRLinalgTransforms) 를 #include 로 가져와 in-tree runOnOperation()
// 과 *동일한 절차* 로 호출한다.  단 legality 술어는 in-tree 에서 file-static
// (:24-31) 이라 export 되지 않으므로, 그 2줄을 exported building block
// (OpTrait::hasElementwiseMappableTraits, MLIRIR) 으로 동일하게 구성한다 —
// 변환 알고리즘이 아니라 ConversionTarget 구성용 술어이며 원본과 줄 단위 동일.
//
// 핵심 학습 포인트:
//   - 본 cycle 첫 dialect-conversion driver pass (#0001~#0004 는 모두 greedy).
//     greedy 와 달리 (a) ConversionTarget 으로 "끝나야 할 상태" 를 선언하고
//     (b) applyPartialConversion 이 illegal op 만 root 로 잡아 pattern 을
//     적용하며 (c) 변환 후에도 illegal op 이 남으면 pass 가 *실패* 한다.
//     단 이 pass 의 target 은 술어의 부정으로 정의되므로 pattern 이 success
//     를 반환하는 한 잔존 illegal 은 구조적으로 생기지 않는다.
//   - pattern 은 op 종류를 안 본다 (MatchAnyOpTypeTag) — *trait 기반* 매치.
//     arith.addf/mulf/cmpf, math.exp 등 ElementwiseMappable trait 을 가진
//     모든 op 이 ranked tensor operand 일 때 일괄 linalg.generic 으로 바뀐다.
//   - body 재생성 트릭: tensor 버전 op 이름 그대로 scalar 버전 op 을 body 에
//     생성 (Elementwise+Scalarizable trait 이 "scalar 에서도 같은 이름으로
//     동작" 을 보증). attribute 도 op->getAttrs() 로 그대로 전달 (cmpf 의
//     predicate 등).
//   - DPS outputs 채우기 (getOrCreateOperandsMatchingResultTypes :46-73):
//     result type == 어떤 operand type 이면 그 operand 를 init 으로 재사용
//     (덮어쓸 대상일 뿐 값은 안 읽음), 다르면 (예: cmpf 의 i1 결과)
//     tensor.empty 생성 — dynamic dim 은 첫 operand 에서 tensor.dim 으로 추출.
//   - 발화 조건이 "모든 operand 가 ranked tensor" (all_of) — scalar 가 섞인
//     broadcast 형태나 unranked tensor, vector operand 는 대상 외 (negative).
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1642)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h" // OpTrait::hasElementwiseMappableTraits (:1502)
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h" // ConversionTarget + applyPartialConversion

using namespace mlir;

namespace {

// in-tree ElementwiseToLinalg.cpp:24-31 과 동일한 legality 술어.
// (in-tree 에선 file-static 이라 export 안 됨 — exported building block 인
//  OpTrait::hasElementwiseMappableTraits 로 같은 2줄을 구성.)
static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;
  return llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

struct MyElementwiseToLinalgPass
    : public PassWrapper<MyElementwiseToLinalgPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyElementwiseToLinalgPass)

  StringRef getArgument() const final { return "my-elementwise-to-linalg"; }

  StringRef getDescription() const final {
    return "Replicate in-tree convert-elementwise-to-linalg: rewrite "
           "ElementwiseMappable ops on ranked tensors into linalg.generic via "
           "populateElementwiseToLinalgConversionPatterns + ConversionTarget "
           "(markUnknownOpDynamicallyLegal) + applyPartialConversion. "
           "[#0005 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:24 — dependentDialects = linalg, memref.
  // (pattern 이 만드는 tensor.empty/tensor.dim 의 TensorDialect 는
  //  LinalgDialect 의 dependent dialect 로 따라 로드된다.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }

  // in-tree ElementwiseToLinalg.cpp:128-141 과 한 줄 한 줄 동일한 절차.
  // (옵션 없음 — in-tree pass 도 옵션이 없다.)
  void runOnOperation() override {
    auto *op = getOperation();
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isElementwiseMappableOpOnRankedTensors(op);
    });

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyElementwiseToLinalgPass() {
  return std::make_unique<MyElementwiseToLinalgPass>();
}

} // namespace linalgtransform
