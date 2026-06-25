//===- HelloInspectPass.cpp - linalg structured op inspect pass --*- C++ -*-===//
//
// #0002 [M0-2] hello-inspect-structured-op.
//
// 함수 안의 모든 linalg structured op 를 walk 하면서, 각 op 가 무엇으로
// 구성되는지를 stdout 으로 출력하는 read-only inspect pass.  IR 은 변경하지
// 않는다.
//
// 학습 목적:  linalg structured op 는 affine.for nest 처럼 *명시적 loop* 로
// 표현되지 않는다.  대신 세 가지 메타데이터로 iteration space 와 데이터
// 접근을 *선언적으로* 기술한다:
//
//   1. indexing_maps   : 각 operand(input/output) 가 iteration space 의
//                        어느 좌표를 읽고/쓰는지를 AffineMap 으로 기술.
//   2. iterator_types  : 각 loop 차원이 parallel 인지 reduction 인지.
//   3. (static) loop ranges : iteration space 의 각 차원 크기(정적일 때).
//   + payload region   : 한 점에서 수행할 스칼라 연산 (block).
//
// 즉 structured op = "indexing_maps + iterator_types + payload" 이며, 이
// pass 는 그 세 요소를 op 별로 끄집어내 보여준다.  이것이 affine dialect 의
// "explicit loop + affine.load/store" 표현과 대비되는 linalg 의 핵심이다.
//
// 베이스 클래스로 OperationPass<func::FuncOp> 를 선택한 이유:
//   - linalg op 는 func body 안에 살며, "이 함수 안의 structured op 들을
//     훑는다" 라는 작업 단위가 func scope 에 자연스럽다.
//   - per-func pass 라 여러 함수가 있어도 함수 경계가 출력에 드러난다.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct HelloInspectPass
    : public PassWrapper<HelloInspectPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloInspectPass)

  // CLI 옵션 이름 (`--my-hello-inspect`)
  StringRef getArgument() const final { return "my-hello-inspect"; }

  // `--help` 에 보일 한 줄 설명
  StringRef getDescription() const final {
    return "Inspect each linalg structured op: print indexing_maps, "
           "iterator_types and static loop ranges (read-only, IR unchanged). "
           "[#0002 linalg-transform-mlir 학습용]";
  }

  // pass 가 의존하는 dialect 들을 미리 로드 (IR 을 만들지는 않지만 linalg/func
  // 타입·인터페이스를 다루므로 명시).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    llvm::outs() << "===== [my-hello-inspect] func @" << fn.getName()
                 << " =====\n";

    unsigned idx = 0;
    fn.walk([&](linalg::LinalgOp op) {
      printLinalgOp(op, idx++);
    });

    if (idx == 0)
      llvm::outs() << "  (no linalg structured op in this function)\n";

    llvm::outs() << "===== [my-hello-inspect] " << idx
                 << " structured op(s) inspected in @" << fn.getName()
                 << " =====\n\n";
  }

private:
  // 하나의 linalg structured op 를 사람이 읽기 좋게 출력한다.
  void printLinalgOp(linalg::LinalgOp op, unsigned n) {
    Operation *raw = op.getOperation();

    llvm::outs() << "  [" << n << "] op = " << raw->getName().getStringRef()
                 << "\n";

    // 1) iterator_types: parallel / reduction 분류 + 총 loop 수
    SmallVector<utils::IteratorType> iters = op.getIteratorTypesArray();
    llvm::outs() << "      iterator_types (" << iters.size() << " loops): [";
    for (size_t i = 0; i < iters.size(); ++i) {
      if (i)
        llvm::outs() << ", ";
      llvm::outs() << utils::stringifyIteratorType(iters[i]);
    }
    llvm::outs() << "]\n";

    // 2) static loop ranges: iteration space 각 차원 크기 (동적이면 '?')
    SmallVector<int64_t, 4> ranges = op.getStaticLoopRanges();
    llvm::outs() << "      static_loop_ranges: [";
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (i)
        llvm::outs() << ", ";
      if (ShapedType::isDynamic(ranges[i]))
        llvm::outs() << "?";
      else
        llvm::outs() << ranges[i];
    }
    llvm::outs() << "]\n";

    // 3) indexing_maps: operand 별로, 어느 iteration 좌표를 읽/쓰는지.
    //    inputs 가 먼저, 그다음 outputs 순서로 들어있다.
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    unsigned numInputs = op.getNumDpsInputs();
    llvm::outs() << "      indexing_maps (" << maps.size() << " operands = "
                 << numInputs << " inputs + "
                 << (maps.size() - numInputs) << " outputs):\n";
    for (size_t i = 0; i < maps.size(); ++i) {
      const char *kind = (i < numInputs) ? "in " : "out";
      llvm::outs() << "        operand[" << i << "] (" << kind << "): ";
      maps[i].print(llvm::outs());
      llvm::outs() << "\n";
    }

    // 4) payload region 의 스칼라 op 들 (한 iteration 점에서 수행되는 계산).
    //    region 이 비어있을 수도 있으므로 방어적으로 처리.
    Block *body = op.getBlock();
    if (body) {
      llvm::outs() << "      payload (region block has " << body->getArguments().size()
                   << " block args, body ops:";
      bool any = false;
      for (Operation &inner : body->getOperations()) {
        llvm::outs() << " " << inner.getName().getStringRef();
        any = true;
      }
      if (!any)
        llvm::outs() << " <empty>";
      llvm::outs() << ")\n";
    }
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<OperationPass<func::FuncOp>> createHelloInspectPass() {
  return std::make_unique<HelloInspectPass>();
}

} // namespace linalgtransform
