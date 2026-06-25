//===- MyInlineScalarOperands.cpp - in-tree inline-scalar pass 재현 -*- C++ -*-===//
//
// #0003 [linalg pass 해부 cycle] linalg-inline-scalar-operands 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/InlineScalarOperands.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgInlineScalarOperandsPass          InlineScalarOperands.cpp:103-115
//       └─ runOnOperation()                   InlineScalarOperands.cpp:108-114
//            ├─ populateInlineConstantOperandsPatterns(patterns)
//            │                                InlineScalarOperands.cpp:95-99
//            │    └─ patterns.add<InlineScalarOperands>(ctx)   (pattern 1개)
//            │         InlineScalarOperands : OpRewritePattern<GenericOp>
//            │                                InlineScalarOperands.cpp:34-90
//            │         └─ matchAndRewrite(GenericOp, PatternRewriter&)  :36-89
//            │              ├─ hasPureTensorSemantics() 아니면 bail    :38-39
//            │              ├─ DPS input 중 indexing map 이
//            │              │  AffineMap::isConstant() (AffineMap.cpp:377-379,
//            │              │  모든 result 가 AffineConstantExpr; empty 도 true
//            │              │  → rank-0 operand 포함) 인 것을 scalar 로 수집 :44-52
//            │              ├─ scalar 없으면 bail                       :54-55
//            │              ├─ scalar input 을 제외한 새 GenericOp 생성  :63-65
//            │              ├─ cloneRegionBefore 로 body 복제           :66-67
//            │              └─ 각 scalar 마다 (역순) body 선두에
//            │                 arith.constant index* + tensor.extract 삽입,
//            │                 해당 block arg RAUW 후 eraseArgument     :73-85
//            └─ applyPatternsAndFoldGreedily(op, patterns)
//                                             InlineScalarOperands.cpp:113 (greedy)
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateInlineConstantOperandsPatterns() (선언 Transforms.h:1722,
// 정의 InlineScalarOperands.cpp:95, lib MLIRLinalgTransforms) 를 #include 로
// 가져와 in-tree runOnOperation() 과 *동일한 절차* 로 호출한다.
//
// 핵심 학습 포인트:
//   - "scalar operand" 판정 = indexing map 의 *모든 result 가 상수*
//     (AffineMap::isConstant).  두 부류가 걸린다:
//       (a) rank-0 tensor<f32> — map `(d0,...) -> ()`, result 0개 (all_of
//           empty → true).  tensor.extract %t[] 로 inline.
//       (b) 상수 인덱스 접근 — map `(d0) -> (0)` 처럼 iteration 변수와 무관한
//           위치 고정 접근.  tensor.extract %t[%c0] 로 inline.
//     즉 "rank-0 이냐 size-1 이냐" 가 아니라 *map 의 상수성* 이 기준이다.
//   - 변환 효과: loop-invariant 한 read 를 generic 의 operand/indexing_maps
//     목록에서 제거하고 body 안 tensor.extract 로 끌어들임 → operand 수,
//     indexing_maps 수, block arg 수가 모두 줄어 후속 fusion/단순화가 쉬워진다.
//   - tensor 전용: hasPureTensorSemantics() 체크 때문에 memref 버전 generic
//     에는 발화하지 않는다 (negative case).
//   - DPS init (output) 은 map 이 상수여도 후보가 아니다 — getDpsInputOperands
//     순회 + isDpsInput 체크 (:44-46).
//   - in-tree pass def (Passes.td:85-90) 는 anchor 없는 Pass<"...">, 옵션 0개,
//     dependentDialects = linalg 뿐 — pattern 이 만드는 arith/tensor op 는
//     LinalgDialect 의 dependent dialect 로 따라 로드된다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1722)
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily

using namespace mlir;

namespace {

struct MyInlineScalarOperandsPass
    : public PassWrapper<MyInlineScalarOperandsPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyInlineScalarOperandsPass)

  StringRef getArgument() const final { return "my-inline-scalar-operands"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-inline-scalar-operands: inline scalar "
           "(constant-indexing-map) operands of linalg.generic as "
           "tensor.extract in the body via "
           "populateInlineConstantOperandsPatterns + greedy driver. "
           "[#0003 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:87-89 — dependentDialects = ["linalg::LinalgDialect"]
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  // in-tree InlineScalarOperands.cpp:108-114 와 한 줄 한 줄 동일한 절차.
  // (옵션 없음 — in-tree pass 도 옵션이 없다.)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateInlineConstantOperandsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyInlineScalarOperandsPass() {
  return std::make_unique<MyInlineScalarOperandsPass>();
}

} // namespace linalgtransform
