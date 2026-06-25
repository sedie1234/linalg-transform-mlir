//===- MyDetensorize.cpp - in-tree linalg-detensorize 재현 -------*- C++ -*-===//
//
// #0008 [linalg pass 해부 cycle] linalg-detensorize 재현.
//
// in-tree 원본: mlir/lib/Dialect/Linalg/Transforms/Detensorize.cpp
//
//   호출 체인 (in-tree, 파일:라인은 LLVM 19.1.7 기준):
//     LinalgDetensorizePass                       Detensorize.cpp:162-575
//       (def: Passes.td:102-137 — 본 cycle 유일의 **InterfacePass**:
//        InterfacePass<"linalg-detensorize", "FunctionOpInterface">.
//        anchor 가 op 이름이 아니라 *interface* — canScheduleOn 이
//        opName.hasInterface<FunctionOpInterface>() 로 판정 (Pass.h:438-440)
//        → func.func 뿐 아니라 모든 function-like op 에 스케줄 가능하고,
//        module 위에는 직접 못 올린다 (예: -pass-pipeline=
//        "builtin.module(func.func(linalg-detensorize))" 로 anchoring).
//        옵션 aggressive-mode (bool, default false, Passes.td:131-136).
//        dependentDialects = [] (Passes.td:104) — 그러나 실제로는
//        cf.br(:490)/tensor.extract(:153)/tensor.from_elements(:40) 를
//        *생성*한다. mlir-opt 류에서 안 깨지는 이유: registerAllExtensions
//        → func::registerInlinerExtension 의 extension 이 FuncDialect 로드
//        시 cf::ControlFlowDialect 를 같이 로드 (InlinerExtension.cpp:83-89).
//        개인 컴파일러 이식 시에는 이 우연에 기대지 말고 cf/tensor 를
//        dependent dialect 로 직접 선언해야 안전 — 본 파일은 그렇게 한다.)
//       └─ runOnOperation()                       Detensorize.cpp:467-574
//            ├─ [0] 빈 body guard                                  :477-478
//            ├─ [1] entry block 보호 트릭                          :484-490
//            │      splitBlock(entryBlock, begin()) + cf.BranchOp 생성 —
//            │      dialect conversion 의 signature 변환이 entry block
//            │      (= 함수 signature) 을 건드리면 함수 type 이 깨지므로,
//            │      본문 전체를 dummy non-entry block 으로 밀어낸다.
//            │      (마지막 [9] 에서 되돌림. detensorize_entry_block.mlir
//            │      테스트가 이 트릭의 회귀 테스트.)
//            ├─ [2] cost model 선택                                :492-500
//            │      aggressiveMode ? AggressiveDetensoringModel(:450-465)
//            │                     : ControlFlowDetectionModel(:254-447)
//            │      → compute(func, typeConverter, opsToDetensor,
//            │                blockArgsToDetensor) 가 "무엇을 풀지" 결정.
//            │      ※ 변환(어떻게)과 선정(무엇을)이 분리된 구조 — greedy
//            │        패턴의 benefit 이 아니라 별도 cost model 객체.
//            ├─ [3] CostModel::computeBranchOpDetensoring          :502-503
//            │      (static, :216-240) blockArgsToDetensor 로부터
//            │      {branch op → 변환할 operand index 집합} 도출.
//            ├─ [4] ConversionTarget legality                      :505-543
//            │      - GenericOp: opsToDetensor 에 없으면 legal     :505-506
//            │      - markUnknownOpDynamicallyLegal                :508-543
//            │        · FunctionOpInterface: 모든 non-entry block 의
//            │          to-detensor blockArg type 이 legal 이면 legal :514-523
//            │        · isNotBranchOpInterfaceOrReturnLikeOp(op) ||
//            │          isLegalForReturnOpTypeConversionPattern(op, tc,
//            │          /*returnOpAlwaysLegal=*/true) → legal      :525-528
//            │          (둘 다 FuncConversions.h:63-73 — export 됨)
//            │        · BranchOpInterface: detensorable 로 지정된 operand
//            │          들이 모두 legal type 이면 legal             :530-540
//            ├─ [5] patterns 3종                                   :545-558
//            │      · DetensorizeGenericOp (OpConversionPattern<GenericOp>,
//            │        :64-93) — generic 의 body region 을 그 자리에 inline:
//            │        splitBlock + inlineRegionBefore + replaceOp(op,
//            │        yield operands) + mergeBlocks(opEntryBlock, ...,
//            │        adaptor.getOperands()=scalar 화된 operand) + 잔여
//            │        yield 제거. tensor 연산 → scalar 연산으로 "껍데기만
//            │        벗기는" 변환의 전부.
//            │      · FunctionNonEntryBlockConversion
//            │        (OpInterfaceConversionPattern<FunctionOpInterface>,
//            │        :97-134) — non-entry block 들의 signature 를
//            │        SignatureConversion 으로 재작성 (blockArgsToDetensor
//            │        에 든 인자만 convertType, 나머지는 그대로) :110-126.
//            │      · populateBranchOpInterfaceTypeConversionPattern
//            │        (patterns, typeConverter, shouldConvertBranchOperand)
//            │        :557-558 — **export 된 populate 함수** (선언
//            │        FuncConversions.h:44-47, 정의 FuncConversions.cpp,
//            │        lib MLIRFuncTransforms). branch operand 를 변환
//            │        대상 index 만 골라 legalize.
//            ├─ [6] driver: applyFullConversion                    :560-562
//            │      — 본 cycle 첫 **FULL** conversion (#0005 는 partial).
//            │      full = target 기준 illegal op 이 하나라도 남으면 실패.
//            ├─ [7] 후처리 greedy                                  :564-568
//            │      FromElementsOp::getCanonicalizationPatterns (TensorOps
//            │      .cpp:1248-1251 — ExtractElementFromIndexCast 1개) +
//            │      applyPatternsAndFoldGreedily. 실효 정리는 패턴보다
//            │      greedy 의 내장 folding/단순화가 수행:
//            │      · ExtractOp::fold — extract(from_elements(x)) → x
//            │        (TensorOps.cpp:1147-1165)
//            │      · FromElementsOp::fold — 상수 → dense<..> 상수
//            │        (TensorOps.cpp:1199-1203)
//            │      · region simplification — 모든 pred 가 같은 값을 넘기는
//            │        redundant blockArg 제거 (negative 입력에서도 발생!)
//            └─ [8]=[9] dummy entry 청소                           :570-573
//                   eraseOp(branch) + mergeBlocks(postEntryBlock, entryBlock)
//
//   type 변환 규칙: DetensorizeTypeConverter            Detensorize.cpp:136-159
//     · canBeDetensored (:51-53): hasRank() && rank==0 — **0-d tensor 만**.
//     · 0-d TensorType → elementType (:143-148), 그 외 type 은 그대로 (:139).
//     · targetMaterialization (:151-154): tensor → scalar 가 필요한 자리에
//       tensor.extract %t[] 삽입.
//     · source/argumentMaterialization (:31-42, :156-157): scalar → tensor 가
//       필요한 자리(예: 함수 return)에 tensor.from_elements 삽입.
//     extract/from_elements 쌍은 [7] 의 fold 로 상쇄 — materialization 은
//     "경계 보정용 임시 op" 이고 최종 IR 에는 경계에만 남는다.
//
//   cost model 상세:
//     · shouldBeDetensored (:55-61): GenericOp 이고 **모든** operand type 이
//       typeConverter 기준 illegal(=0-d tensor) 일 때만. (all_of — 입력
//       하나라도 rank>0 이면 탈락.)
//     · ControlFlowDetectionModel::compute (:256-446): cf.cond_br/cf.br 의
//       operand 들을 seed 로 worklist 양방향 탐색.
//       전방(:301-313): 값이 terminator 로 후속 block 에 넘어가면 그 blockArg
//         추가, 값의 user 들의 result 추가.
//       후방(:315-405): blockArg 면 blockArgsToDetensor 에 넣고 predecessor
//         가 넘기는 operand 추적(:321-362; pred terminator 가
//         BranchOpInterface 가 아니면 **전부 포기** :339-345); GenericOp 이면
//         shouldBeDetensored 검사 후 opsToDetensor + inputs 추적(:369-388);
//         tensor.from_elements 는 [7] 에서 정리되므로 skip(:390-398);
//         scalar op 면 operand 추적(:400-405).
//       마무리(:412-445): detensor 하지 않기로 한 generic 이 feed 하는
//         blockArg 를 집합에서 제외 (불일치 방지).
//     · AggressiveDetensoringModel::compute (:452-464): walk 로 모든
//       shouldBeDetensored generic + 모든 non-entry blockArg 를 무조건 수집.
//
// 본 파일은 알고리즘을 재구현하지 않는다 — 단, 이 pass 는 #0001~#0007 과
// 달리 자기 전용 populate* 함수가 없고 핵심 pattern/cost model 이 모두
// Detensorize.cpp 의 **file-local(익명 namespace) class** 라 export 되지
// 않는다. 따라서:
//   (a) export 된 building block 은 전부 #include 로 가져와 호출 —
//       populateBranchOpInterfaceTypeConversionPattern,
//       isNotBranchOpInterfaceOrReturnLikeOp,
//       isLegalForReturnOpTypeConversionPattern (MLIRFuncTransforms),
//       applyFullConversion / applyPatternsAndFoldGreedily (driver),
//       tensor::FromElementsOp::getCanonicalizationPatterns,
//       TypeConverter/OpConversionPattern 인프라.
//   (b) export 안 된 file-local 조각 (sourceMaterializationCallback,
//       canBeDetensored, shouldBeDetensored, DetensorizeGenericOp,
//       FunctionNonEntryBlockConversion, DetensorizeTypeConverter,
//       CostModel/ControlFlowDetectionModel/AggressiveDetensoringModel,
//       runOnOperation 본문) 은 Detensorize.cpp:31-574 를 **줄 단위 동일
//       이식**(verbatim port) — 개인 컴파일러 이식 시나리오 그대로.
// 옵션 aggressive-mode 도 in-tree 와 같은 이름·같은 기본값(false)으로 노출.
//
//===----------------------------------------------------------------------===//

#include "MyPasses/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h" // cf::BranchOp (:490)
#include "mlir/Dialect/Func/Transforms/FuncConversions.h" // populateBranchOpInterfaceTypeConversionPattern 등 (:44-73)
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h" // InterfacePass<FunctionOpInterface> anchor
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h" // ConversionTarget + applyFullConversion
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // 후처리 greedy (:566)
#include "llvm/Support/CommandLine.h"
#include <iterator>
#include <memory>
#include <utility>

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// 이하 Detensorize.cpp:31-159 의 file-local 조각 verbatim port
//===----------------------------------------------------------------------===//

// in-tree Detensorize.cpp:31-42 와 동일.
// scalar → tensor 방향 보정(함수 return 등 변환 경계): tensor.from_elements.
static Value sourceMaterializationCallback(OpBuilder &builder, Type type,
                                           ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (isa<TensorType>(inputType))
    return nullptr;

  // A detensored value is converted back by creating a new tensor from its
  // element(s).
  return builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({}, inputType), inputs[0]);
}

namespace {

// in-tree Detensorize.cpp:51-53 과 동일. "detensor 가능" 판정: 0-d tensor 만.
bool canBeDetensored(TensorType tensorType) {
  return tensorType.hasRank() && tensorType.getRank() == 0;
}

// in-tree Detensorize.cpp:55-61 과 동일. GenericOp 이고 모든 operand 가
// illegal(=0-d tensor) type 일 때만 detensor 대상.
bool shouldBeDetensored(Operation *op, TypeConverter typeConverter) {
  GenericOp genericOp = dyn_cast_or_null<GenericOp>(op);
  return genericOp &&
         llvm::all_of(genericOp->getOpOperands(), [&](OpOperand &opOperand) {
           return !typeConverter.isLegal(opOperand.get().getType());
         });
}

/// in-tree Detensorize.cpp:64-93 (DetensorizeGenericOp) 과 동일.
/// linalg.generic 의 body 를 둘러싼 block 에 그대로 inline 해 scalar 연산만
/// 남긴다. adaptor.getOperands() 는 typeConverter 의 targetMaterialization
/// (tensor.extract) 을 거친 scalar 값들이다.
class MyDetensorizeGenericOp : public OpConversionPattern<GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *originalBlock = op->getBlock();

    // Gather some information about the op before inlining its region.
    Block *opEntryBlock = &*op.getRegion().begin();
    YieldOp yieldOp = dyn_cast<YieldOp>(op.getRegion().back().getTerminator());

    // Split the op's region before the op. This way, we have a clear insertion
    // point in which the op can be inlined.
    Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));
    rewriter.inlineRegionBefore(op.getRegion(), newBlock);
    // Now that op's region is inlined, the operands of its YieldOp are mapped
    // to the materialized target values. Therefore, we can replace the op's
    // uses with those of its YielOp's operands.
    rewriter.replaceOp(op, yieldOp->getOperands());

    // No need for these intermediate blocks, merge them into 1.
    rewriter.mergeBlocks(opEntryBlock, originalBlock, adaptor.getOperands());
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(&*Block::iterator(yieldOp));

    return success();
  }
};

/// in-tree Detensorize.cpp:97-134 (FunctionNonEntryBlockConversion) 과 동일.
/// 함수의 non-entry block 들의 signature(인자 type)를 재작성. entry block 은
/// 함수 type 과 묶여 있으므로 건드리지 않는다 (drop_begin(region, 1)).
struct MyFunctionNonEntryBlockConversion
    : public OpInterfaceConversionPattern<FunctionOpInterface> {
  MyFunctionNonEntryBlockConversion(MLIRContext *ctx, TypeConverter &converter,
                                    DenseSet<BlockArgument> blockArgsToDetensor)
      : OpInterfaceConversionPattern(converter, ctx),
        blockArgsToDetensor(std::move(blockArgsToDetensor)) {}

  LogicalResult
  matchAndRewrite(FunctionOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    Region &region = op.getFunctionBody();

    for (Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region, 1))) {
      TypeConverter::SignatureConversion conversion(
          /*numOrigInputs=*/block.getNumArguments());

      for (BlockArgument blockArgument : block.getArguments()) {
        int idx = blockArgument.getArgNumber();

        if (blockArgsToDetensor.count(blockArgument))
          conversion.addInputs(idx, {getTypeConverter()->convertType(
                                        block.getArgumentTypes()[idx])});
        else
          conversion.addInputs(idx, {block.getArgumentTypes()[idx]});
      }

      rewriter.applySignatureConversion(&block, conversion, getTypeConverter());
    }

    rewriter.finalizeOpModification(op);
    return success();
  }

private:
  const DenseSet<BlockArgument> blockArgsToDetensor;
};

/// in-tree Detensorize.cpp:136-159 (DetensorizeTypeConverter) 와 동일.
class MyDetensorizeTypeConverter : public TypeConverter {
public:
  MyDetensorizeTypeConverter() {
    addConversion([](Type type) { return type; });

    // A TensorType that can be detensored, is converted to the underlying
    // element type.
    addConversion([](TensorType tensorType) -> Type {
      if (canBeDetensored(tensorType))
        return tensorType.getElementType();

      return tensorType;
    });

    // A tensor value is detensoried by extracting its element(s).
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<tensor::ExtractOp>(loc, inputs[0], ValueRange{});
    });

    addSourceMaterialization(sourceMaterializationCallback);
    addArgumentMaterialization(sourceMaterializationCallback);
  }
};

//===----------------------------------------------------------------------===//
// pass 본체 — in-tree Detensorize.cpp:162-575 (LinalgDetensorize) 와 동일 절차
//===----------------------------------------------------------------------===//

struct MyDetensorizePass
    : public PassWrapper<MyDetensorizePass, InterfacePass<FunctionOpInterface>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyDetensorizePass)

  MyDetensorizePass() = default;
  MyDetensorizePass(const MyDetensorizePass &pass) : PassWrapper(pass) {}

  // in-tree Passes.td:131-136 과 동일한 이름·타입·기본값의 옵션.
  Option<bool> aggressiveMode{
      *this, "aggressive-mode",
      llvm::cl::desc("Detensorize all ops that qualify for detensoring along "
                     "with branch operands and basic-block arguments."),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "my-detensorize"; }

  StringRef getDescription() const final {
    return "Replicate in-tree linalg-detensorize: rewrite 0-d-tensor "
           "linalg.generic ops (and the control flow carrying them) into "
           "primitive scalar ops via DetensorizeGenericOp + "
           "FunctionNonEntryBlockConversion + "
           "populateBranchOpInterfaceTypeConversionPattern, cost model "
           "(ControlFlowDetectionModel | AggressiveDetensoringModel), "
           "applyFullConversion, then FromElementsOp canonicalization via "
           "greedy driver. [#0008 linalg-transform-mlir 학습용]";
  }

  // in-tree 는 dependentDialects = [] (Passes.td:104) 이지만 실제로 cf.br /
  // tensor.extract / tensor.from_elements 를 생성하며, mlir-opt 에서는 func
  // inliner extension 이 cf 를 우연히 로드해 줘서 안 깨질 뿐이다
  // (InlinerExtension.cpp:83-89). 이식판은 생성하는 op 의 dialect 를 정직하게
  // 선언한다 (출력 IR 에는 영향 없음 — 로드 시점만 보장).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, tensor::TensorDialect>();
  }

  // in-tree Detensorize.cpp:168-241 (CostModel) 과 동일.
  class CostModel {
  public:
    virtual ~CostModel() = default;

    /// cost model 의 출력 2개:
    /// - opsToDetensor: detensor 할 linalg op 집합.
    /// - blockArgsToDetensor: detensor 된 값이 BB 경계를 넘을 때 함께 type
    ///   변환되어야 하는 non-entry block argument 집합.
    virtual void compute(FunctionOpInterface func,
                         MyDetensorizeTypeConverter typeConverter,
                         DenseSet<Operation *> &opsToDetensor,
                         DenseSet<BlockArgument> &blockArgsToDetensor) = 0;

    /// blockArgsToDetensor 로부터 {branch op → detensor 할 operand index 집합}
    /// 을 계산. (in-tree :216-240 과 동일.)
    static DenseMap<Operation *, DenseSet<int>> computeBranchOpDetensoring(
        const DenseSet<BlockArgument> &blockArgsToDetensor) {
      DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;

      for (auto blockArgumentElem : blockArgsToDetensor) {
        Block *block = blockArgumentElem.getOwner();

        for (PredecessorIterator pred = block->pred_begin();
             pred != block->pred_end(); ++pred) {
          BranchOpInterface terminator =
              dyn_cast<BranchOpInterface>((*pred)->getTerminator());
          auto blockOperands =
              terminator.getSuccessorOperands(pred.getSuccessorIndex());

          if (blockOperands.empty() ||
              blockOperands.isOperandProduced(blockArgumentElem.getArgNumber()))
            continue;

          detensorableBranchOps[terminator].insert(
              blockOperands.getOperandIndex(blockArgumentElem.getArgNumber()));
        }
      }

      return detensorableBranchOps;
    }
  };

  /// in-tree Detensorize.cpp:254-447 (ControlFlowDetectionModel) 과 동일.
  /// cf.br/cf.cond_br 의 operand 에서 출발해 use-def chain 을 양방향으로 걸어
  /// "control flow 에 관여하면서 detensor 가능한 component" 를 발견한다.
  class ControlFlowDetectionModel : public CostModel {
  public:
    void compute(FunctionOpInterface func,
                 MyDetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      SmallVector<Value> workList;

      func->walk([&](cf::CondBranchOp condBr) {
        llvm::append_range(workList, condBr.getOperands());
      });

      func->walk([&](cf::BranchOp br) {
        llvm::append_range(workList, br.getOperands());
      });

      DenseSet<Value> visitedValues;
      DenseSet<Operation *> visitedOps;

      // 값이 terminator 를 통해 block 을 "탈출" 하면 후속 block 의 대응
      // 인자를 workList 에 추가. (in-tree :276-293)
      auto updateWorkListWithSuccessorArguments =
          [&](Value value, BranchOpInterface terminator) {
            if (!terminator)
              return;

            for (auto operandIdx :
                 llvm::seq<unsigned>(0, terminator->getOperands().size())) {
              Value operand = terminator->getOperand(operandIdx);

              if (operand == value) {
                auto succBlockArg =
                    terminator.getSuccessorBlockArgument(operandIdx);

                if (succBlockArg && !blockArgsToDetensor.count(*succBlockArg))
                  workList.push_back(*succBlockArg);
              }
            }
          };

      while (!workList.empty()) {
        Value currentItem = workList.pop_back_val();

        if (!visitedValues.insert(currentItem).second)
          continue;

        // 1   - 전방 탐색 (in-tree :301-313):
        // 1.1 - currentItem 이 후속 block 으로 탈출하면 그 인자 추가.
        updateWorkListWithSuccessorArguments(
            currentItem, dyn_cast<BranchOpInterface>(
                             currentItem.getParentBlock()->getTerminator()));

        // 1.2 - user 들의 result 를 추가해 component 의 나머지를 발견.
        for (auto *user : currentItem.getUsers())
          llvm::append_range(workList, user->getResults());

        // 2   - 후방 탐색:
        // 2.1 - blockArg 이면 (non-entry 한정) detensor 집합에 넣고
        //       predecessor 가 넘기는 operand 를 추적. (in-tree :315-362)
        if (dyn_cast<BlockArgument>(currentItem)) {
          BlockArgument currentItemBlockArgument =
              cast<BlockArgument>(currentItem);
          Block *ownerBlock = currentItemBlockArgument.getOwner();

          // Function arguments are not detensored/converted.
          if (&*ownerBlock->getParent()->begin() == ownerBlock)
            continue;

          // This inner-block argument is involved in control-flow, it should
          // be detensored.
          blockArgsToDetensor.insert(currentItemBlockArgument);

          for (PredecessorIterator pred = ownerBlock->pred_begin();
               pred != ownerBlock->pred_end(); ++pred) {
            BranchOpInterface predTerminator =
                dyn_cast<BranchOpInterface>((*pred)->getTerminator());

            // pred terminator 가 BranchOpInterface 가 아니면 함수 전체 포기.
            // (in-tree :339-345 TODO 그대로)
            if (!predTerminator) {
              opsToDetensor.clear();
              blockArgsToDetensor.clear();
              return;
            }

            auto ownerBlockOperands =
                predTerminator.getSuccessorOperands(pred.getSuccessorIndex());

            if (ownerBlockOperands.empty() ||
                ownerBlockOperands.isOperandProduced(
                    currentItemBlockArgument.getArgNumber()))
              continue;

            // For each predecessor, add the value it passes to that argument
            // to workList to find out how it's computed.
            workList.push_back(
                ownerBlockOperands[currentItemBlockArgument.getArgNumber()]);
          }

          continue;
        }

        Operation *currentItemDefiningOp = currentItem.getDefiningOp();

        if (!visitedOps.insert(currentItemDefiningOp).second)
          continue;

        // 2.2 - GenericOp 이 정의한 값이면 opsToDetensor 에 넣고 inputs 추적.
        //       (in-tree :369-388)
        if (auto genericOp = dyn_cast<GenericOp>(currentItemDefiningOp)) {
          // The op was encountered already, no need to inspect it again.
          if (opsToDetensor.count(genericOp))
            continue;

          // The op should not be detensored, give up on it but continue with
          // discovering the rest of the control-flow component.
          if (!shouldBeDetensored(genericOp, typeConverter)) {
            continue;
          }

          opsToDetensor.insert(genericOp);
          llvm::append_range(workList, genericOp.getInputs());
          continue;
        }

        // 2.3 - tensor.from_elements 는 마지막 canonicalization 이 정리하므로
        //       여기서는 통과. (in-tree :390-398)
        if (isa<tensor::FromElementsOp>(currentItemDefiningOp))
          continue;

        // 2.4 - scalar op 이면 operand 들을 추적. (in-tree :400-405)
        if (llvm::all_of(
                currentItemDefiningOp->getResultTypes(),
                [&](Type resultType) { return resultType.isIntOrFloat(); }))
          llvm::append_range(workList, currentItemDefiningOp->getOperands());
      }

      // 2.2 에서 포기한 generic 이 feed 하는 blockArg 는 detensor 대상에서
      // 제외해 모순을 막는다. (in-tree :412-445)
      DenseSet<BlockArgument> blockArgsToRemove;

      for (auto &blockArg : blockArgsToDetensor) {
        Block *block = blockArg.getParentBlock();

        // For the potentially detensorable block argument, find the
        // correpsonding operands in predecessor blocks.
        for (PredecessorIterator pred = block->pred_begin();
             pred != block->pred_end(); ++pred) {
          BranchOpInterface terminator =
              dyn_cast<BranchOpInterface>((*pred)->getTerminator());
          auto blockOperands =
              terminator.getSuccessorOperands(pred.getSuccessorIndex());

          if (blockOperands.empty() ||
              blockOperands.isOperandProduced(blockArg.getArgNumber()))
            continue;

          Operation *definingOp =
              blockOperands[blockArg.getArgNumber()].getDefiningOp();

          // If the operand is defined by a GenericOp that will not be
          // detensored, then do not detensor the corresponding block argument.
          if (isa_and_nonnull<GenericOp>(definingOp) &&
              opsToDetensor.count(definingOp) == 0) {
            blockArgsToRemove.insert(blockArg);
            break;
          }
        }
      }

      for (auto &blockArg : blockArgsToRemove) {
        blockArgsToDetensor.erase(blockArg);
      }
    }
  };

  /// in-tree Detensorize.cpp:450-465 (AggressiveDetensoringModel) 와 동일.
  /// detensor 가능한 것 전부: 모든 shouldBeDetensored generic + 모든
  /// non-entry blockArg.
  class AggressiveDetensoringModel : public CostModel {
  public:
    void compute(FunctionOpInterface func,
                 MyDetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      func->walk([&](GenericOp genericOp) {
        if (shouldBeDetensored(genericOp, typeConverter))
          opsToDetensor.insert(genericOp);
      });

      for (Block &block : llvm::drop_begin(func.getFunctionBody(), 1))
        for (BlockArgument blockArgument : block.getArguments())
          blockArgsToDetensor.insert(blockArgument);
    }
  };

  // in-tree Detensorize.cpp:467-574 와 한 줄 한 줄 동일한 절차.
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    MyDetensorizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    DenseSet<Operation *> opsToDetensor;
    DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;
    DenseSet<BlockArgument> blockArgsToDetensor;
    FunctionOpInterface funcOp = getOperation();

    if (funcOp.getFunctionBody().empty())
      return;

    // [1] entry block 보호: 본문을 dummy non-entry block 으로 밀어내
    // signature 변환이 함수 type 을 깨지 못하게 한다. (in-tree :480-490)
    IRRewriter rewriter(funcOp->getContext());
    Block *entryBlock = &funcOp.getFunctionBody().front();
    Block *postEntryBlock =
        rewriter.splitBlock(entryBlock, entryBlock->begin());
    rewriter.setInsertionPointToStart(entryBlock);
    auto branch =
        rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), postEntryBlock);

    // [2] cost model 선택. (in-tree :492-500)
    if (aggressiveMode.getValue()) {
      AggressiveDetensoringModel costModel;
      costModel.compute(funcOp, typeConverter, opsToDetensor,
                        blockArgsToDetensor);
    } else {
      ControlFlowDetectionModel costModel;
      costModel.compute(funcOp, typeConverter, opsToDetensor,
                        blockArgsToDetensor);
    }

    // [3] branch operand 변환 대상 계산. (in-tree :502-503)
    detensorableBranchOps =
        CostModel::computeBranchOpDetensoring(blockArgsToDetensor);

    // [4] ConversionTarget legality. (in-tree :505-543)
    target.addDynamicallyLegalOp<GenericOp>(
        [&](GenericOp op) { return !opsToDetensor.count(op); });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // A function is legal if all of its non-entry blocks are legal. We
      // don't legalize the entry block (i.e. the function's signature)
      // since detensoring can't happen along external calling convention
      // boundaries, which we conservatively approximate as all function
      // signatures.
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        Region &body = funcOp.getFunctionBody();
        return llvm::all_of(llvm::drop_begin(body, 1), [&](Block &block) {
          return !llvm::any_of(
              blockArgsToDetensor, [&](BlockArgument blockArgument) {
                return blockArgument.getOwner() == &block &&
                       !typeConverter.isLegal(blockArgument.getType());
              });
        });
      }

      if (isNotBranchOpInterfaceOrReturnLikeOp(op) ||
          isLegalForReturnOpTypeConversionPattern(op, typeConverter,
                                                  /*returnOpAlwaysLegal*/ true))
        return true;

      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        if (!detensorableBranchOps.count(branchOp))
          return true;

        for (auto operandIdx : detensorableBranchOps[branchOp])
          if (!typeConverter.isLegal(
                  branchOp->getOperand(operandIdx).getType()))
            return false;

        return true;
      }

      return false;
    });

    // [5] patterns 3종. (in-tree :545-558)
    patterns.add<MyDetensorizeGenericOp>(typeConverter, context);
    patterns.add<MyFunctionNonEntryBlockConversion>(context, typeConverter,
                                                    blockArgsToDetensor);
    // Since non-entry block arguments get detensorized, we also need to
    // update the control flow inside the function to reflect the correct
    // types.
    auto shouldConvertBranchOperand = [&](BranchOpInterface branchOp,
                                          int operandIdx) -> bool {
      return detensorableBranchOps.count(branchOp) &&
             detensorableBranchOps[branchOp].count(operandIdx);
    };

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter,
                                                   shouldConvertBranchOperand);

    // [6] driver: FULL conversion. (in-tree :560-562)
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();

    // [7] 후처리 greedy: from_elements/extract 쌍 정리 + region 단순화.
    // (in-tree :564-568)
    RewritePatternSet canonPatterns(context);
    tensor::FromElementsOp::getCanonicalizationPatterns(canonPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(canonPatterns))))
      signalPassFailure();

    // [8] dummy entry block 청소. (in-tree :570-573)
    rewriter.eraseOp(branch);
    rewriter.mergeBlocks(postEntryBlock, entryBlock);
  }
};

} // namespace

namespace linalgtransform {

std::unique_ptr<mlir::Pass> createMyDetensorizePass() {
  return std::make_unique<MyDetensorizePass>();
}

} // namespace linalgtransform
