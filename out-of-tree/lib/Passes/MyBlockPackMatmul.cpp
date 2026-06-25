//===- MyBlockPackMatmul.cpp - in-tree block-pack-matmul 재현 -*- C++ -*-===//
//
// #0010 [linalg pass 해부 cycle] linalg-block-pack-matmul 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/BlockPackMatmul.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgBlockPackMatmul                       BlockPackMatmul.cpp:279-307
//       (def: Passes.td:139-196 — Pass<"linalg-block-pack-matmul">,
//        anchor 없음 = op-agnostic.
//        dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"]
//        (Passes.td:172).
//        옵션 8개 (Passes.td:173-195):
//          ListOption blockFactors / "block-factors" — (mb, nb, kb)
//          Option allowPadding / "allow-padding" / default true
//          ListOption mnkPaddedSizesNextMultipleOf / "mnk-padded-multiples"
//          ListOption mnkOrder / "mnk-order" — (M,N,K) 순서 permutation
//          Option lhsTransposeOuterBlocks / default false — [MB][KB]→[KB][MB]
//          Option lhsTransposeInnerBlocks / default false — [mb][kb]→[kb][mb]
//          Option rhsTransposeOuterBlocks / default true  — [KB][NB]→[NB][KB]
//          Option rhsTransposeInnerBlocks / default true  — [kb][nb]→[nb][kb]
//        → RHS 기본값 2개가 true 라서 기본 결과가 mmt4d 형
//          [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb] 이 된다.)
//       └─ runOnOperation()                      BlockPackMatmul.cpp:283-306
//            ├─ ControlBlockPackMatmulFn controlFn = lambda (:287-301)
//            │    ★ 옵션 8개가 흐르는 유일한 경로 — pass 의 tablegen 옵션
//            │    멤버들을 BlockPackMatmulOptions(Transforms.h:1185-1211) 로
//            │    복사해 돌려주는 closure. mnkOrder 만 "비어 있으면 기본
//            │    {0,1,2} 유지" 분기 (:294-295). 이 controlFn 은 pattern 이
//            │    op 마다 blockPackMatmul 안에서 호출한다 (op별 정책 훅).
//            ├─ populateBlockPackMatmulPatterns(patterns, controlFn)   :303
//            │    선언 Transforms.h:1742-1743 / 정의 BlockPackMatmul.cpp:310-320
//            │    └─ BlockPackMatmul<OpTy> pattern 7 instantiation:
//            │         GenericOp(전문화 :236-276) + named 6종 = MatmulOp,
//            │         BatchMatmulOp, MatmulTransposeAOp,
//            │         BatchMatmulTransposeAOp, MatmulTransposeBOp,
//            │         BatchMatmulTransposeBOp (primary template :217-234)
//            │       · named 용 matchAndRewrite (:223-230) = 단순 위임 →
//            │         linalg::blockPackMatmul(rewriter, linalgOp, controlFn)
//            │       · GenericOp 전문화 matchAndRewrite (:244-272) = 먼저
//            │         isaContractionOpInterface 확인 (:247) + indexing maps
//            │         가 {(i,k),(k,j)}, {(k,i),(k,j)}, {(i,k),(j,k)} → (i,j)
//            │         세 형태만 허용 (:261-263, "simple matmuls only") 후
//            │         동일 위임 (:267-268)
//            │       └─ linalg::blockPackMatmul                 :138-214
//            │            (선언 Transforms.h:1241-1243)
//            │            1. hasPureBufferSemantics → fail (:141-142)
//            │               — tensor semantics 전용 (memref 입력은 no-op)
//            │            2. controlPackMatmul(linalgOp) → options (:144-146)
//            │            3. blockFactors.size() != 3 → fail (:148-149)
//            │               ★ block-factors 미지정 시 전체 no-op 의 근거:
//            │               옵션 기본값이 빈 리스트 → 모든 pattern 이
//            │               "require 3 tile factors" 로 matchFailure →
//            │               greedy 가 IR 을 그대로 둔다.
//            │            4. !allowPadding 이면 validateFullTilesOnDims
//            │               (:44-86) — TilingInterface::getIterationDomain
//            │               의 각 (m,n,k) range 가 tile 로 나눠떨어지는지
//            │               (:154-159). 아니면 fail.
//            │            5. packMatmulGreedily(rewriter, op, mnkTiles,
//            │               mnkPaddedSizesNextMultipleOf, mnkOrder)
//            │               (:169-171) — Transforms.cpp:768-898:
//            │                 a. inferContractionDims (LinalgInterfaces.cpp
//            │                    :372) 로 (m,n,k) iterator 위치 추론
//            │                 b. named op 이면 generalizeNamedOp → generic
//            │                    (Transforms.cpp:829-835)
//            │                 c. interchangeGenericOp 로 (k,m,n) 을
//            │                    most-minor iterator 로 정규화 (:837-848)
//            │                 d. mnk-padded-multiples 지정 시 tile 크기를
//            │                    affine ceilDiv*s 로 올림 (:874-888)
//            │                 e. linalg::pack (Transforms.cpp:480-610) —
//            │                    operand 3개에 tensor.pack (나눠떨어지지
//            │                    않으면 padding_value 포함), packed
//            │                    linalg.generic 재생성(7D: 4 outer + mb,nb,kb
//            │                    minor), 결과에 tensor.unpack 1개
//            │            6. inferContractionDims(packedLinalgOp) (:180-183)
//            │            7. transposePackedMatmul × 2 (:190-211, 본체 :89-135)
//            │               — LHS 는 contractDims->m + lhsTranspose* 옵션,
//            │               RHS 는 contractDims->k + rhsTranspose* 옵션으로:
//            │               operandMap 의 outer/inner block dim 위치가 이미
//            │               transposed 인지 판정(:107-110)해 원하는 설정과
//            │               다를 때만 perm={1,0} (:114-119), batch 등 선행
//            │               outer dim 은 identity 로 보존(:123-128) 후
//            │               packTranspose (Transforms.cpp:677-755) 호출 —
//            │               tensor.pack 의 outer_dims_perm/inner_dims_perm
//            │               재작성 + generic 의 해당 indexing map 도 같은
//            │               permutation 으로 치환.
//            └─ applyPatternsAndFoldGreedily(op, std::move(patterns))   :304
//                 → **greedy driver**, GreedyRewriteConfig 기본값.
//                 실패 시 signalPassFailure (:305).
//
// 본 파일은 알고리즘을 재구현하지 않는다 — in-tree 가 export 하는
// linalg::populateBlockPackMatmulPatterns (MLIRLinalgTransforms) 를
// #include 로 가져와 in-tree runOnOperation() 과 *동일한 절차* 로 호출한다.
// 옵션 8개도 in-tree 와 같은 이름·타입·기본값으로 노출한다.
//
// 핵심 학습 포인트:
//   - 이 pass 는 "한 개의 큰 transform 함수(blockPackMatmul) + 옵션 정책
//     closure(ControlBlockPackMatmulFn)" 구조. pattern 은 op 종류 필터일
//     뿐이고 실제 결정(블록 크기·transpose 여부)은 controlFn 이 op 마다
//     돌려주는 BlockPackMatmulOptions 가 전담한다 — 이식 시 controlFn 만
//     바꾸면 op별 휴리스틱(예: 크기에 따라 다른 블록 인자)을 줄 수 있다.
//   - 변환 자체는 기존 빌딩블록 3개의 합성: packMatmulGreedily(= generalize
//     + interchange + pack) → inferContractionDims → packTranspose×2.
//   - tensor.pack/unpack 은 layout 변환의 *서술* 이며, 기본 옵션의 결과
//     generic 은 linalg.mmt4d 와 같은 접근 패턴([MB][NB][mb][nb] +=
//     [MB][KB][mb][kb] * [NB][KB][nb][kb]) 이 된다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // populate 선언 (:1742)
#include "mlir/Dialect/Tensor/IR/Tensor.h" // dependentDialects (Passes.td:172)
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

struct MyBlockPackMatmulPass
    : public PassWrapper<MyBlockPackMatmulPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyBlockPackMatmulPass)

  MyBlockPackMatmulPass() = default;
  MyBlockPackMatmulPass(const MyBlockPackMatmulPass &pass)
      : PassWrapper(pass) {}

  // in-tree Passes.td:173-195 와 동일한 이름·타입·기본값의 옵션 8개.
  ListOption<int64_t> blockFactors{
      *this, "block-factors",
      llvm::cl::desc("Block factors (mb, nb, kb) for relayout")};
  Option<bool> allowPadding{*this, "allow-padding",
                            llvm::cl::desc("Allow packing padding"),
                            llvm::cl::init(true)};
  ListOption<int64_t> mnkPaddedSizesNextMultipleOf{
      *this, "mnk-padded-multiples",
      llvm::cl::desc("Next multiples of the packing sizes")};
  ListOption<int64_t> mnkOrder{
      *this, "mnk-order",
      llvm::cl::desc("Permutation of matmul (M, N, K) dimensions order")};
  Option<bool> lhsTransposeOuterBlocks{
      *this, "lhs-transpose-outer-blocks",
      llvm::cl::desc("Transpose LHS outer block layout [MB][KB] -> [KB][MB]"),
      llvm::cl::init(false)};
  Option<bool> lhsTransposeInnerBlocks{
      *this, "lhs-transpose-inner-blocks",
      llvm::cl::desc("Transpose LHS inner block layout [mb][kb] -> [kb][mb]"),
      llvm::cl::init(false)};
  Option<bool> rhsTransposeOuterBlocks{
      *this, "rhs-transpose-outer-blocks",
      llvm::cl::desc("Transpose RHS outer block layout [KB][NB] -> [NB][KB]"),
      llvm::cl::init(true)};
  Option<bool> rhsTransposeInnerBlocks{
      *this, "rhs-transpose-inner-blocks",
      llvm::cl::desc("Transpose RHS inner block layout [kb][nb] -> [nb][kb]"),
      llvm::cl::init(true)};

  StringRef getArgument() const final { return "my-block-pack-matmul"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-block-pack-matmul: pack matmul ops into "
           "blocked 4D layout (tensor.pack ×3 + packed linalg.generic + "
           "tensor.unpack) via populateBlockPackMatmulPatterns → "
           "blockPackMatmul (packMatmulGreedily + packTranspose×2) + greedy "
           "driver. [#0010 linalg-transform-mlir 학습용]";
  }

  // in-tree Passes.td:172 — dependentDialects = linalg, tensor.
  // (pack 이 만드는 arith.constant(padding zero)/affine.apply 는 linalg
  //  dialect 의 dependent dialect 로 따라 로드된다.)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  // in-tree BlockPackMatmul.cpp:283-306 과 한 줄 한 줄 동일한 절차.
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(&getContext());

    linalg::ControlBlockPackMatmulFn controlFn =
        [&](linalg::LinalgOp op) -> linalg::BlockPackMatmulOptions {
      linalg::BlockPackMatmulOptions options;
      options.blockFactors = SmallVector<int64_t>{*blockFactors};
      options.allowPadding = allowPadding;
      options.mnkPaddedSizesNextMultipleOf =
          SmallVector<int64_t>{*mnkPaddedSizesNextMultipleOf};
      if (!mnkOrder.empty())
        options.mnkOrder = SmallVector<int64_t>{*mnkOrder};
      options.lhsTransposeOuterBlocks = lhsTransposeOuterBlocks;
      options.lhsTransposeInnerBlocks = lhsTransposeInnerBlocks;
      options.rhsTransposeOuterBlocks = rhsTransposeOuterBlocks;
      options.rhsTransposeInnerBlocks = rhsTransposeInnerBlocks;
      return options;
    };

    linalg::populateBlockPackMatmulPatterns(patterns, controlFn);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyBlockPackMatmulPass() {
  return std::make_unique<MyBlockPackMatmulPass>();
}

} // namespace linalgtransform
