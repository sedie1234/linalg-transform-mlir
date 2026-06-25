//===- Passes.h - linalg-transform-mlir 학습용 본인 pass 등록 진입점 --*- C++ -*-===//
//
// 본 헤더는 out-of-tree 워크스페이스의 모든 본인 pass 를 외부에 노출한다.
// my-mlir-opt 는 main() 진입 시 registerMyPasses() 한 번만 호출하면
// 모든 사용자 pass 가 mlir-opt 기능에 자동으로 추가된다.
//
// (2026-06-12 재시작) 본 워크스페이스의 현 학습 cycle 은 *in-tree linalg pass*
// 의 내부 코드 구성(pass → pattern → 핵심 함수 체인)과 적용 전후 IR 변화를
// 파악하는 것이다.  각 cycle 의 본인 pass 는 in-tree pass 가 쓰는 함수
// (populate*Patterns / 핵심 transform 함수) 를 out-of-tree 에서 가져와 호출해
// in-tree pass 를 재현한다.
//
//===----------------------------------------------------------------------===//

#ifndef MY_PASSES_PASSES_H
#define MY_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace linalgtransform {

//===----------------------------------------------------------------------===//
// 개별 pass factory (각 pass 의 .cpp 가 정의)
//===----------------------------------------------------------------------===//

/// 함수 안의 모든 linalg structured op 를 walk 하며, 각 op 의 핵심 구성요소
/// (indexing_maps + iterator_types + static loop ranges) 를 stdout 에 사람이
/// 읽기 좋은 형식으로 출력하는 read-only inspect pass.  IR 은 변경하지 않는다.
/// 인프라 스모크용.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createHelloInspectPass();

/// #0001 in-tree linalg-generalize-named-ops 재현 (my-generalize-named-ops).
/// 체인: runOnOperation → populateLinalgNamedOpsGeneralizationPatterns
///       → LinalgGeneralizationPattern(matchAndRewrite) → generalizeNamedOp
///       → applyPatternsAndFoldGreedily.  (Generalization.cpp:89-98 동일 절차)
std::unique_ptr<mlir::Pass> createMyGeneralizeNamedOpsPass();

/// #0002 in-tree linalg-specialize-generic-ops 재현 (my-specialize-generic-ops).
/// 체인: runOnOperation → populateLinalgGenericOpsSpecializationPatterns
///       → LinalgSpecializationPattern(matchAndRewrite) → specializeGenericOp
///       (copy→fill→exp→binary→contraction 순 idiom 판정)
///       → applyPatternsAndFoldGreedily.  (Specialize.cpp:322-333 동일 절차)
std::unique_ptr<mlir::Pass> createMySpecializeGenericOpsPass();

/// #0003 in-tree linalg-inline-scalar-operands 재현 (my-inline-scalar-operands).
/// 체인: runOnOperation → populateInlineConstantOperandsPatterns
///       → InlineScalarOperands(matchAndRewrite: 상수 indexing map 인 DPS input
///       을 새 GenericOp 에서 제거하고 body 에 tensor.extract 로 inline)
///       → applyPatternsAndFoldGreedily.  (InlineScalarOperands.cpp:108-114 동일 절차)
std::unique_ptr<mlir::Pass> createMyInlineScalarOperandsPass();

/// #0004 in-tree linalg-named-op-conversion 재현 (my-named-op-conversion).
/// 체인: runOnOperation → populateLinalgNamedOpConversionPatterns
///       → SimplifyDepthwiseConv(Q)Op(matchAndRewrite)
///       → matchAndReplaceDepthwiseConv(multiplier=1 인 depthwise_conv_2d_
///       nhwc_hwcm(_q) 를 collapse_shape + *_hwc(_q) + expand_shape 로 좁힘)
///       → applyPatternsAndFoldGreedily.  (NamedOpConversions.cpp:151-157 동일 절차)
std::unique_ptr<mlir::Pass> createMyNamedOpConversionPass();

/// #0005 in-tree convert-elementwise-to-linalg 재현 (my-elementwise-to-linalg).
/// 체인: runOnOperation → populateElementwiseToLinalgConversionPatterns
///       → ConvertAnyElementwiseMappableOpOnRankedTensors(matchAndRewrite:
///       ElementwiseMappable trait + 전 operand ranked tensor 인 op 을
///       identity map × parallel 의 linalg.generic 으로 치환, body 에 같은
///       이름의 scalar op 재생성) → ConversionTarget(markUnknownOpDynamically
///       Legal) + applyPartialConversion.  (ElementwiseToLinalg.cpp:128-141
///       동일 절차 — 본 cycle 첫 dialect-conversion driver.)
std::unique_ptr<mlir::Pass> createMyElementwiseToLinalgPass();

/// #0006 in-tree linalg-fuse-elementwise-ops 재현 (my-fuse-elementwise-ops).
/// 체인: runOnOperation → populateElementwiseOpsFusionPatterns(+FoldReshape
///       OpsByExpansion/+canonicalization/+ConstantFold, controlFn=producer
///       hasOneUse) → FuseElementwiseOps(matchAndRewrite: areElementwiseOps
///       Fusable → fuseElementwiseOps — indexing map 합성으로 producer/consumer
///       generic 을 한 generic 으로 병합, region splice) →
///       applyPatternsAndFoldGreedily(top-down).
///       (ElementwiseOpFusion.cpp:2133-2163 동일 절차)
std::unique_ptr<mlir::Pass> createMyFuseElementwiseOpsPass();

/// #0007 in-tree linalg-fold-unit-extent-dims 재현 (my-fold-unit-extent-dims).
/// 체인: runOnOperation → populateFoldUnitExtentDimsPatterns(strategy 분기:
///       기본 ReassociativeReshape ↔ 옵션 use-rank-reducing-slices 시
///       ExtractInsertSlice) → DropUnitDims(matchAndRewrite →
///       linalg::dropUnitDims: inversePermutation(concatAffineMaps) 로
///       one-trip iteration dim 검출 → dim 을 const 0 치환 + operand
///       collapse_shape/extract_slice + 결과 expand_shape/insert_slice)
///       +DropPadUnitDims +RankReduced{Extract,Insert}SliceOp(reshape 모드만)
///       +populateMoveInitOperandsToInputPattern →
///       applyPatternsAndFoldGreedily.  (DropUnitDims.cpp:822-834 동일 절차)
std::unique_ptr<mlir::Pass> createMyFoldUnitExtentDimsPass();

/// #0008 in-tree linalg-detensorize 재현 (my-detensorize).
/// 체인: runOnOperation(InterfacePass<FunctionOpInterface>) → entry block
///       보호(splitBlock+cf.br) → cost model(기본 ControlFlowDetectionModel:
///       cf.br/cond_br operand 에서 use-def 양방향 탐색 ↔ aggressive-mode:
///       모든 0-d generic) → DetensorizeGenericOp(body inline) +
///       FunctionNonEntryBlockConversion(blockArg type 강하) +
///       populateBranchOpInterfaceTypeConversionPattern →
///       applyFullConversion → FromElementsOp canonicalization greedy
///       → dummy entry 복원.  (Detensorize.cpp:467-574 동일 절차)
std::unique_ptr<mlir::Pass> createMyDetensorizePass();

/// #0009 in-tree convert-linalg-to-{loops,affine-loops,parallel-loops} 3종
/// 재현 (my-linalg-to-loops-trio, 옵션 mode=scf|affine|parallel).
/// 체인: runOnOperation → [in-tree lowerLinalgToLoopsImpl<LoopTy> 와 동일]
///       LinalgRewritePattern(matchAndRewrite: buffer-semantics LinalgOp →
///       linalgOpTo{Loops,AffineLoops,ParallelLoops} = linalgOpToLoopsImpl
///       <LoopTy>: createLoopRanges → GenerateLoopNest<LoopTy>::doit →
///       emitScalarImplementation(makeCanonicalAffineApplies 로 indexing_map
///       → load/store index, region clone-inline) → linalg.index → iv 치환)
///       + DimOp/AffineApplyOp canonicalization + FoldAffineOp →
///       applyPatternsAndFoldGreedily.  (Loops.cpp:314-325 동일 절차)
std::unique_ptr<mlir::Pass> createMyLinalgToLoopsTrioPass();

/// #0010 in-tree linalg-block-pack-matmul 재현 (my-block-pack-matmul).
/// 체인: runOnOperation → controlFn(옵션 8개→BlockPackMatmulOptions closure)
///       → populateBlockPackMatmulPatterns(BlockPackMatmul<OpTy> 7종:
///       generic 전문화 + named matmul 6종) → matchAndRewrite →
///       linalg::blockPackMatmul(tensor semantics·blockFactors==3 확인 →
///       packMatmulGreedily = generalize + interchange(k,m,n most-minor) +
///       pack(tensor.pack×3 + packed generic + tensor.unpack) →
///       transposePackedMatmul×2 = packTranspose 로 LHS/RHS outer·inner
///       block layout 조정) → applyPatternsAndFoldGreedily.
///       (BlockPackMatmul.cpp:283-306 동일 절차)
std::unique_ptr<mlir::Pass> createMyBlockPackMatmulPass();

//===----------------------------------------------------------------------===//
// 일괄 등록 진입점
//===----------------------------------------------------------------------===//

void registerMyPasses();

} // namespace linalgtransform

#endif // MY_PASSES_PASSES_H
