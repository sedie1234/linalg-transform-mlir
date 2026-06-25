//===- main.cpp - my-mlir-opt standalone driver ----------------*- C++ -*-===//
//
// linalg-transform-mlir 학습용 standalone driver.
//
// 표준 mlir-opt 기능을 그대로 제공한다:
//   - --help / --pass-pipeline / --mlir-print-* / FileCheck 친화
//   - 모든 in-tree dialect 와 pass 등록 (registerAllDialects / registerAllPasses)
//   - 모든 in-tree extension 등록 (registerAllExtensions)  ← 핵심.
//     이로써 transform dialect 의 op interface impl (linalg 의
//     transform.structured.* op 등) 과 --transform-interpreter pass 가
//     이 driver 에서 실제로 동작한다.
//   - 본 워크스페이스의 사용자 pass (linalgtransform::registerMyPasses())
//
// affine-mlir 버전이 갖고 있던 T1 PassManager-직접구성 hook
// (buildT1PassManagerFromFlags / setPassPipelineSetupFn) 은 제거했다.
// 이 워크스페이스는 transform dialect 의 *명시적 schedule IR* 로 변환을
// 기술하므로, C++ 쪽 PassManager hack 대신 표준 드라이버 + transform
// interpreter 경로를 쓴다.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "MyPasses/Passes.h"

int main(int argc, char **argv) {
  // 모든 in-tree pass 등록 (canonicalize, cse, transform-interpreter 등)
  mlir::registerAllPasses();

  // 모든 in-tree dialect + extension 등록.
  // registerAllExtensions 가 transform dialect 용 op-interface impl 들을
  // 붙여줘야 transform.structured.tile_using_for 같은 linalg transform op 이
  // 인터프리터에서 정상 동작한다.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  // 본 워크스페이스의 사용자 pass 일괄 등록 (#0002 hello-inspect ...)
  linalgtransform::registerMyPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "my-mlir-opt (linalg+transform 학습 driver)\n", registry));
}
