//===- PassRegistration.cpp - 본인 pass 일괄 등록 -------------*- C++ -*-===//
//
// my-mlir-opt 가 main() 에서 한 번만 호출하는 진입점.
// 새 pass 를 추가할 때는 아래 registerPass 한 줄만 추가하면 된다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace linalgtransform {

void registerMyPasses() {
  // PassRegistration<ConcretePass> 템플릿은 ConcretePass 가 default-constructible
  // 일 때 가장 간단하지만, 우리는 익명 namespace 안에 pass class 를 두고
  // factory function 만 외부에 노출했다 (헤더 깨끗하게 유지). 따라서
  // factory function 을 그대로 받는 ::mlir::registerPass() 자유 함수를 쓴다.

  // hello-inspect — linalg structured op 의 indexing_maps / iterator_types /
  // static loop ranges 를 print (read-only). 인프라 스모크.
  ::mlir::registerPass(createHelloInspectPass);

  // (2026-06-12 재시작) in-tree linalg pass 내부구조 학습 cycle 의 pass 들이
  // 여기에 한 줄씩 추가된다.

  // #0001 — in-tree linalg-generalize-named-ops 재현. named op → generic.
  ::mlir::registerPass(createMyGeneralizeNamedOpsPass);

  // #0002 — in-tree linalg-specialize-generic-ops 재현. generic → named op
  // (copy/fill/exp/add/sub/mul/div/matmul 변형). #0001 의 역방향.
  ::mlir::registerPass(createMySpecializeGenericOpsPass);

  // #0003 — in-tree linalg-inline-scalar-operands 재현. 상수 indexing map
  // (rank-0 포함) DPS input 을 body 안 tensor.extract 로 inline.
  ::mlir::registerPass(createMyInlineScalarOperandsPass);

  // #0004 — in-tree linalg-named-op-conversion 재현. depthwise conv 의
  // multiplier(M)=1 이면 *_hwcm(_q) → *_hwc(_q) + collapse/expand_shape.
  ::mlir::registerPass(createMyNamedOpConversionPass);

  // #0005 — in-tree convert-elementwise-to-linalg 재현. ElementwiseMappable
  // trait + ranked tensor operand 인 op (arith.addf 등) → linalg.generic.
  // 본 cycle 첫 dialect-conversion driver (applyPartialConversion).
  ::mlir::registerPass(createMyElementwiseToLinalgPass);

  // #0006 — in-tree linalg-fuse-elementwise-ops 재현. elementwise
  // linalg.generic producer-consumer fusion + reshape-by-expansion 전파 +
  // fill/splat/outs fold + canonicalization + constant fold 를
  // greedy(top-down) 고정점까지.
  ::mlir::registerPass(createMyFuseElementwiseOpsPass);

  // #0007 — in-tree linalg-fold-unit-extent-dims 재현. broadcasting 용
  // unit-extent dim 을 indexing map 역사상으로 검출해 const-0 치환 +
  // collapse/expand (기본) 또는 extract/insert_slice (옵션) 로 제거.
  ::mlir::registerPass(createMyFoldUnitExtentDimsPass);

  // #0008 — in-tree linalg-detensorize 재현. 0-d tensor 의 linalg.generic 과
  // 그것을 나르는 제어흐름(blockArg/branch operand)을 scalar 로 강하.
  // 본 cycle 유일의 InterfacePass(FunctionOpInterface) + applyFullConversion.
  ::mlir::registerPass(createMyDetensorizePass);

  // #0009 — in-tree convert-linalg-to-{loops,affine-loops,parallel-loops}
  // 3종 재현 (한 template 기계의 3 instantiation → mode 옵션 1개로 통합).
  // buffer-semantics linalg op → 명시적 loop nest + load/compute/store.
  ::mlir::registerPass(createMyLinalgToLoopsTrioPass);

  // #0010 — in-tree linalg-block-pack-matmul 재현. matmul 류(named 6종 +
  // generic contraction)를 blocked 4D layout 으로 packing (tensor.pack×3 +
  // packed generic + tensor.unpack, 기본 옵션이면 mmt4d 형).
  // block-factors 미지정 시 no-op (require 3 tile factors).
  ::mlir::registerPass(createMyBlockPackMatmulPass);
}

} // namespace linalgtransform
