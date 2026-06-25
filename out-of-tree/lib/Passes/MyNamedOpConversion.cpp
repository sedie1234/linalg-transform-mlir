//===- MyNamedOpConversion.cpp - in-tree named-op-conversion 재현 -*- C++ -*-===//
//
// #0004 [linalg pass 해부 cycle] linalg-named-op-conversion 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/NamedOpConversions.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgNamedOpConversionPass             NamedOpConversions.cpp:145-158
//       (def: Passes.td:80-83 — anchor 없는 Pass<"linalg-named-op-conversion">,
//        옵션 0개, dependentDialects = ["linalg::LinalgDialect",
//        "tensor::TensorDialect"])
//       └─ runOnOperation()                   NamedOpConversions.cpp:151-157
//            ├─ populateLinalgNamedOpConversionPatterns(patterns)
//            │       선언 Transforms.h:1711 / 정의 NamedOpConversions.cpp:161-165
//            │    └─ patterns.add<SimplifyDepthwiseConvOp,
//            │                    SimplifyDepthwiseConvQOp>(ctx)   (pattern 2개)
//            │
//            │       SimplifyDepthwiseConvOp
//            │         : OpRewritePattern<DepthwiseConv2DNhwcHwcmOp>   :104-122
//            │         └─ matchAndRewrite                              :108-121
//            │              └─ matchAndReplaceDepthwiseConv(op, input, kernel,
//            │                     /*iZp=*/nullptr, /*kZp=*/nullptr, init,
//            │                     strides, dilations, rewriter)       :118-120
//            │       SimplifyDepthwiseConvQOp
//            │         : OpRewritePattern<DepthwiseConv2DNhwcHwcmQOp>  :124-143
//            │         └─ matchAndRewrite (iZp/kZp = DPS input 2,3)    :128-142
//            │
//            │       matchAndReplaceDepthwiseConv (static 핵심 함수)   :35-101
//            │         시그니처: static LogicalResult
//            │           matchAndReplaceDepthwiseConv(Operation *, Value input,
//            │             Value kernel, Value iZp, Value kZp, Value init,
//            │             Attribute stride, Attribute dilation,
//            │             PatternRewriter &)
//            │         ├─ hasPureTensorSemantics() 아니면 bail          :40-43
//            │         ├─ kernel/init/result 가 RankedTensorType 아니면 bail
//            │         │                                                :47-51
//            │         ├─ kernelTy.getDimSize(3) != 1 이면 bail
//            │         │   (= multiplier 차원이 *정적으로* 1 일 때만;
//            │         │    dynamic '?' 는 kDynamic != 1 이라 bail)     :53-54
//            │         ├─ kernel collapse_shape [[0],[1],[2,3]]
//            │         │   HWCM(h,w,c,1) → HWC(h,w,c)                  :56-63
//            │         ├─ init collapse_shape [[0],[1],[2],[3,4]]
//            │         │   NHWCM(n,h,w,c,1) → NHWC(n,h,w,c)            :65-74
//            │         ├─ TypeSwitch 로 새 named op 생성:
//            │         │   DepthwiseConv2DNhwcHwcmOp  → DepthwiseConv2DNhwcHwcOp
//            │         │   DepthwiseConv2DNhwcHwcmQOp → DepthwiseConv2DNhwcHwcQOp
//            │         │   (그 외 → nullptr → failure)                 :76-93
//            │         ├─ getPrunedAttributeList(op) (Utils.h:368-374) 로
//            │         │   discardable attr (예: _someattr) 만 골라 새 op 에
//            │         │   재부착 — 정의된 attr(strides 등)·memoized indexing
//            │         │   maps 는 제외                                 :80,86,94-95
//            │         └─ replaceOpWithNewOp<tensor::ExpandShapeOp>
//            │             (같은 reassociation 으로 결과를 원래 5-D 로 복원)
//            │                                                          :97-99
//            └─ applyPatternsAndFoldGreedily(op, std::move(patterns))   :155
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// populateLinalgNamedOpConversionPatterns() (선언 Transforms.h:1711,
// 정의 NamedOpConversions.cpp:161, lib MLIRLinalgTransforms) 를 #include 로
// 가져와 in-tree runOnOperation() 과 *동일한 절차* 로 호출한다.
//
// 핵심 학습 포인트:
//   - "named → named 정규화" pass 인데 19.1.7 시점의 실체는 단 한 종류:
//     depthwise conv 2D 의 channel multiplier 차원(M) 이 정적으로 1 이면
//     *_hwcm(_q) → *_hwc(_q) 로 좁히는 것 (비양자화/양자화 2 pattern).
//   - named op 자체는 rank 가 고정돼 있어 (hwcm 커널=4-D, 출력=5-D) 차원을
//     "지우는" 변환이 named op 수준에선 불가 — 그래서 tensor.collapse_shape
//     (입력측) / tensor.expand_shape (결과측) 로 감싸 rank 를 맞춘다.
//     dependentDialects 에 tensor::TensorDialect 가 들어가는 이유.
//   - multiplier=1 판정은 *정적* shape 기준 (getDimSize(3) != 1 → bail).
//     `?x...` 처럼 M 이 dynamic 이면 실제 값이 1 이어도 변환 안 함 (negative).
//   - memref 버전(named conv on memrefs)은 hasPureTensorSemantics() 에서
//     bail — collapse/expand 가 tensor op 이기 때문 (negative).
//   - discardable attribute 보존: getPrunedAttributeList 가 op 정의 attr
//     (strides/dilations/operandSegmentSizes 등) 을 뺀 나머지(_someattr 류)만
//     골라 새 op 으로 옮긴다 — 변환 후에도 사용자 메타데이터 유지.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate* 선언 (:1711)
#include "mlir/Dialect/Tensor/IR/Tensor.h" // tensor::TensorDialect (dependent)
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily

using namespace mlir;

namespace {

struct MyNamedOpConversionPass
    : public PassWrapper<MyNamedOpConversionPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyNamedOpConversionPass)

  StringRef getArgument() const final { return "my-named-op-conversion"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-named-op-conversion: canonicalize "
           "depthwise_conv_2d_nhwc_hwcm(_q) with static multiplier==1 into "
           "depthwise_conv_2d_nhwc_hwc(_q) wrapped in tensor.collapse_shape/"
           "expand_shape, via populateLinalgNamedOpConversionPatterns + "
           "greedy driver. [#0004 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:80-83 — dependentDialects =
  //   ["linalg::LinalgDialect", "tensor::TensorDialect"]
  // (pattern 이 tensor.collapse_shape / tensor.expand_shape 를 새로 만들므로
  //  tensor dialect 가 반드시 로드돼 있어야 한다.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  // in-tree NamedOpConversions.cpp:151-157 과 한 줄 한 줄 동일한 절차.
  // (옵션 없음 — in-tree pass 도 옵션이 없다.)
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    linalg::populateLinalgNamedOpConversionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyNamedOpConversionPass() {
  return std::make_unique<MyNamedOpConversionPass>();
}

} // namespace linalgtransform
