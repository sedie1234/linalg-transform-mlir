# linalg-transform-mlir 학습용 본인 pass 라이브러리
#
# 새 pass 를 추가할 때는 본 SOURCES 리스트에 .cpp 한 줄을 추가하고
# PassRegistration.cpp 의 registerMyPasses() 에 등록 한 줄을 추가하면 된다.
#
# (2026-06-12 재시작: in-tree linalg pass 내부구조 학습 cycle.
#  이전 transform-API cycle 의 pass 26종은 archive/2026-06-cycle1-transform-api/
#  out-of-tree-passes/ 에 보관.)

add_mlir_library(MyLinalgTransformPasses
  HelloInspectPass.cpp
  MyBlockPackMatmul.cpp
  MyDetensorize.cpp
  MyElementwiseToLinalg.cpp
  MyFoldUnitExtentDims.cpp
  MyFuseElementwiseOps.cpp
  MyGeneralizeNamedOps.cpp
  MyInlineScalarOperands.cpp
  MyLinalgToLoopsTrio.cpp
  MyNamedOpConversion.cpp
  MySpecializeGenericOps.cpp
  PassRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyPasses

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRLinalgDialect
  MLIRFuncDialect
  # #0004 my-named-op-conversion 이 dependent dialect 로 직접 참조
  # (pattern 이 tensor.collapse_shape/expand_shape 생성).
  MLIRTensorDialect
  # #0005 my-elementwise-to-linalg 이 in-tree Passes.td:24 그대로
  # dependentDialects 에 memref 를 선언하므로 직접 참조.
  MLIRMemRefDialect
  # #0006 my-fuse-elementwise-ops 가 in-tree Passes.td:75-77 그대로
  # dependentDialects 에 affine 을 선언하고 AffineApplyOp 의
  # canonicalization 패턴을 직접 참조.
  MLIRAffineDialect
  # in-tree linalg pass 의 populate*Patterns / 핵심 함수가 사는 lib —
  # 본 cycle 의 모든 pass 가 여기서 함수를 가져와 호출한다.
  MLIRLinalgTransforms
  # applyPatternsAndFoldGreedily 등 pattern driver.
  MLIRTransforms
  # #0008 my-detensorize 가 entry-block 보호 트릭으로 cf.br 을 직접 생성.
  MLIRControlFlowDialect
  # #0008 my-detensorize 가 populateBranchOpInterfaceTypeConversionPattern /
  # isNotBranchOpInterfaceOrReturnLikeOp /
  # isLegalForReturnOpTypeConversionPattern (FuncConversions.h) 을 직접 호출.
  MLIRFuncTransforms
  # #0009 my-linalg-to-loops-trio 가 in-tree Passes.td:42-46/:52-58 그대로
  # dependentDialects 에 scf 를 선언 (scf.for/scf.parallel 생성).
  MLIRSCFDialect
  # #0009 의 FoldAffineOp(Loops.cpp:287-312 복제) 가 arith.constant index
  # 를 직접 생성.
  MLIRArithDialect
)
