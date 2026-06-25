# #0008 detensorize — in-tree `linalg-detensorize` 해부·재현·관찰

in-tree pass `linalg-detensorize` (LLVM 19.1.7,
`mlir/lib/Dialect/Linalg/Transforms/Detensorize.cpp`) 를 해부하고,
out-of-tree pass `my-detensorize`
(`out-of-tree/lib/Passes/MyDetensorize.cpp`) 로 같은 절차를 재현해
byte-diff 로 이식을 검증한 실험.

**Detensoring** = 0-d tensor (`tensor<i32>` 등) 로 박제된 값과 그 위의
`linalg.generic` 연산을 primitive scalar (`i32`) 와 scalar 연산으로 끌어내리는
변환. tensor 기반 프런트엔드(TOSA/HLO 류)가 loop counter·조건 같은 *제어용
스칼라*까지 tensor 로 들고 오는 것을 풀어, CFG(block argument / branch
operand) 레벨에서 scalar 가 흐르게 한다.

## 호출 체인 (해부 결과 — 파일:라인은 LLVM 19.1.7)

```
LinalgDetensorizePass                                  Passes.td:102-137
  = InterfacePass<"linalg-detensorize", "FunctionOpInterface">   ← 본 cycle 유일의 InterfacePass
    옵션: aggressive-mode (bool, default false)        Passes.td:131-136
    dependentDialects = [] (질문거리 — 아래 §특이점)   Passes.td:104
struct LinalgDetensorize : impl::LinalgDetensorizePassBase<…>
                                                       Detensorize.cpp:162-575
└─ runOnOperation()                                    Detensorize.cpp:467-574
     [0] 빈 body guard                                  :477-478
     [1] entry block 보호: splitBlock + cf.br 생성      :484-490
     [2] cost model 선택 (옵션 분기)                    :492-500
         aggressive-mode=false → ControlFlowDetectionModel :254-447
         aggressive-mode=true  → AggressiveDetensoringModel :450-465
         → compute(func, typeConverter, opsToDetensor, blockArgsToDetensor)
     [3] CostModel::computeBranchOpDetensoring          :502-503 (정의 :216-240)
         → {branch op → 변환할 operand index 집합}
     [4] ConversionTarget legality                      :505-543
         · GenericOp: opsToDetensor 에 없으면 legal     :505-506
         · FunctionOpInterface: non-entry block 의 to-detensor
           blockArg 가 모두 legal type 이면 legal       :514-523
         · isNotBranchOpInterfaceOrReturnLikeOp ||
           isLegalForReturnOpTypeConversionPattern(…, true) :525-528
           (FuncConversions.h:63-73, export 됨)
         · BranchOpInterface: 지정 operand 모두 legal type 이면 legal :530-540
     [5] patterns 3종                                   :545-558
         · DetensorizeGenericOp(typeConverter, ctx)     :64-93
         · FunctionNonEntryBlockConversion(ctx, tc, blockArgs) :97-134
         · populateBranchOpInterfaceTypeConversionPattern(
             patterns, tc, shouldConvertBranchOperand)  :557-558
           (선언 FuncConversions.h:44-47, lib MLIRFuncTransforms)
     [6] driver: applyFullConversion                    :560-562  ← FULL conversion
     [7] 후처리: FromElementsOp::getCanonicalizationPatterns
         (TensorOps.cpp:1248-1251) + applyPatternsAndFoldGreedily :564-568
     [8] dummy entry 청소: eraseOp(br) + mergeBlocks    :570-573

type 규칙: DetensorizeTypeConverter                     Detensorize.cpp:136-159
  canBeDetensored = hasRank() && rank==0                :51-53   ← 0-d 만!
  0-d TensorType → elementType                          :143-148
  targetMaterialization  = tensor.extract %t[]          :151-154 (tensor→scalar)
  source/argumentMaterialization = tensor.from_elements :31-42, :156-157 (scalar→tensor)

선정 규칙: shouldBeDetensored                            :55-61
  GenericOp && 모든 operand type 이 illegal(=0-d tensor) (all_of)
```

### cost model 두 개 (무엇을 풀지 — 변환과 분리된 결정)

| | ControlFlowDetectionModel (기본) | AggressiveDetensoringModel (`aggressive-mode`) |
|---|---|---|
| 정의 | Detensorize.cpp:254-447 | Detensorize.cpp:450-465 |
| seed | `cf.cond_br`/`cf.br` 의 operand (:262-268) | `walk` 로 모든 GenericOp |
| 탐색 | use-def chain 양방향: 전방(후속 blockArg·user result :301-313), 후방(blockArg→pred operand :315-362, GenericOp→inputs :369-388, from_elements skip :390-398, scalar op→operands :400-405) | 탐색 없음 — `shouldBeDetensored` 면 전부 (:456-459) |
| blockArg | 탐색 중 만난 non-entry blockArg 만 (:332) | **모든** non-entry blockArg (:461-463) |
| 포기 조건 | pred terminator 가 BranchOpInterface 아니면 함수 전체 포기 (:339-345); detensor 안 될 generic 이 feed 하는 blockArg 사후 제외 (:412-445) | 없음 |
| 효과 | *제어흐름에 관여하는* 0-d generic 만 scalar 화 | 제어흐름 무관 0-d generic 도 scalar 화 |

### 특이점 (이식 시 주의)

1. **유일한 InterfacePass**: anchor 가 op 이름이 아니라 interface.
   `canScheduleOn = opName.hasInterface<FunctionOpInterface>()` (Pass.h:438-440)
   → module 위에 직접 못 올린다. `--linalg-detensorize` 단독 호출은
   `unable to schedule pass` 에러; 반드시
   `-pass-pipeline="builtin.module(func.func(linalg-detensorize))"`.
2. **entry block 보호 트릭** (:484-490, :570-573): dialect conversion 의
   signature 변환이 entry block 을 건드리면 함수 type 이 깨지므로, 본문
   전체를 `splitBlock`+`cf.br` 로 dummy non-entry block 으로 밀어냈다가
   끝에 되돌린다. `detensorize_entry_block.mlir` 가 이 트릭의 회귀 테스트.
3. **dependentDialects = [] 인데 cf.br/tensor.* 를 생성**: mlir-opt 류에서
   안 깨지는 이유는 `registerAllExtensions` → func inliner extension 이
   FuncDialect 로드 시 `cf::ControlFlowDialect` 를 같이 로드해 주기 때문
   (InlinerExtension.cpp:83-89). 개인 컴파일러에 이식할 때는 이 우연에
   기대지 말고 cf/tensor 를 dependent dialect 로 직접 선언할 것 —
   `MyDetensorize.cpp` 는 그렇게 했다 (출력 IR 영향 없음).
4. **populate* 함수가 없는 pass**: 핵심 pattern/cost model 이 전부
   Detensorize.cpp 의 file-local class 라 export 되지 않는다. 재현은
   (a) export 된 building block (`populateBranchOpInterfaceTypeConversionPattern`,
   `isNotBranchOpInterfaceOrReturnLikeOp`,
   `isLegalForReturnOpTypeConversionPattern`, `applyFullConversion`,
   `applyPatternsAndFoldGreedily`, `FromElementsOp::getCanonicalizationPatterns`)
   호출 + (b) file-local 조각의 줄 단위 동일 이식(verbatim port) 으로 구성.
5. **applyFullConversion**: #0005 의 partial 과 달리 full — 변환 후 target
   기준 illegal op 이 남으면 pass 실패. legality 가 cost model 의 결정
   (opsToDetensor/blockArgsToDetensor) 을 그대로 반영하도록 동적으로
   구성되므로, "결정했으면 반드시 끝까지 변환" 을 driver 가 강제한다.

## 코드 단계 ↔ IR 변화 매핑

### 1) `input/while.mlir` → `output/output.while.mlir` (기본 모드, 발화)

0-d tensor 로 표현된 while loop. `cf.cond_br` 의 조건이 `linalg.generic`
(cmpi) 의 결과에서 오고, loop carried 값도 `linalg.generic`(addi) 결과.

| IR 변화 (before → after) | 만든 코드 |
|---|---|
| `^bb1(%0: tensor<i32>)` → `^bb1(%0: i32)` (모든 non-entry blockArg type 강하) | `FunctionNonEntryBlockConversion::matchAndRewrite` :104-130 의 `applySignatureConversion` + cost model 이 채운 `blockArgsToDetensor` |
| `cf.br ^bb1(%farg0 : tensor<i32>)` → `cf.br ^bb1(%extracted_0 : i32)` (branch operand 가 scalar 로) | `populateBranchOpInterfaceTypeConversionPattern` 의 pattern (FuncConversions.cpp) + `computeBranchOpDetensoring` :216-240 이 고른 operand index |
| `%2 = linalg.generic {…cmpi…} -> tensor<i1>` → `%1 = arith.cmpi slt, %0, %extracted : i32` (generic 이 사라지고 body 의 scalar 연산이 그 자리에) | `DetensorizeGenericOp::matchAndRewrite` :67-92 — region inline(`splitBlock`+`inlineRegionBefore`+`mergeBlocks(opEntryBlock,…,adaptor.getOperands())`) |
| 함수 인자 `%arg0: tensor<i32>` 는 그대로, 본문 첫머리에 `%extracted_0 = tensor.extract %arg0[]` 삽입 | `DetensorizeTypeConverter` 의 targetMaterialization :151-154 (entry block=함수 signature 는 불변 — legality :514-523 이 entry 를 제외) |
| return 직전 `%from_elements = tensor.from_elements %0 : tensor<i32>` 삽입, `return %from_elements` | sourceMaterialization :31-42 — 함수 결과 type 은 `tensor<i32>` 로 유지되어야 하므로 scalar 를 재포장 |
| `tensor.empty() : tensor<i1>` (generic 의 outs init) 소멸 | generic 치환 후 user 가 없어져 [7] greedy 의 DCE 가 제거 |
| 원본의 `%3 = tensor.extract %2[]` (조건 추출) 소멸 | generic 결과가 처음부터 scalar 가 되어 extract(from_elements) 쌍이 `ExtractOp::fold` (TensorOps.cpp:1147-1165) 로 상쇄 |
| `cf.cond_br %3, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)` → `cf.cond_br %1, ^bb2, ^bb3` (successor 인자 자체가 소멸) | [7] greedy 의 region simplification — 모든 pred 가 같은 값을 넘기는 redundant blockArg 제거 |

aggressive 모드 출력도 이 입력에서는 동일 (`output.agg.while.mlir` ==
`output.while.mlir`): 모든 0-d generic 이 이미 제어흐름에 관여하므로 두
cost model 의 결정이 일치.

### 2) `input/pure-compute.mlir` — 모드 대조 (기본=비발화 / aggressive=발화)

제어흐름에 관여하지 않는 0-d `linalg.generic`(addf) 하나.

- **기본 모드** (`output.pure-compute.mlir`): **불변**.
  `ControlFlowDetectionModel` 의 seed 는 `cf.br`/`cf.cond_br` 의 operand 인데
  (:262-268) 이 함수에는 branch 가 없어 workList 가 비어 있음 →
  `opsToDetensor = {}` → 모든 op legal → 변환 0. ([1] 의 dummy entry 생성과
  [8] 의 복원이 일어나지만 정확히 원상복구됨을 byte-diff 로 확인.)
- **aggressive 모드** (`output.agg.pure-compute.mlir`):
  `AggressiveDetensoringModel` 이 walk 로 generic 을 수집 (:456-459) →
  `linalg.generic` 1개가 `tensor.extract` ×2 + `arith.addf` +
  `tensor.from_elements` 로 강하. 함수 경계(인자/결과)는 tensor 로 유지 —
  materialization 이 경계 보정만 담당.

```mlir
// aggressive 모드 after (전체):
func.func @detensorable_but_no_cf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %extracted = tensor.extract %arg1[] : tensor<f32>      // targetMaterialization :151-154
  %extracted_0 = tensor.extract %arg0[] : tensor<f32>    // targetMaterialization :151-154
  %0 = arith.addf %extracted_0, %extracted : f32         // DetensorizeGenericOp :67-92 (body inline)
  %from_elements = tensor.from_elements %0 : tensor<f32> // sourceMaterialization :31-42
  return %from_elements : tensor<f32>
}
```

### 3) `input/rank1-negative.mlir` — 진짜 negative (양 모드 비발화)

`tensor<4xf32>` 의 generic 이 제어흐름(`cf.cond_br`→`cf.br`)에 관여하지만
rank 1 → `canBeDetensored` :51-53 이 false → typeConverter 가 type 을
바꾸지 않음(=legal) + `shouldBeDetensored` :55-61 false → 양 모드 모두
generic/blockArg/branch 불변.

단, 출력이 입력과 *완전히* 같지는 않다: `^bb1(%x: tensor<4xf32>)` 의
인자가 사라지고 `%arg0` 직사용으로 바뀌는데, 이는 detensorize 의 변환이
아니라 [7] `applyPatternsAndFoldGreedily` 에 기본 탑재된 **region
simplification** (단일 값만 받는 redundant blockArg 제거) 의 효과다.
"비발화 ≠ 출력 무변화" — 이 pass 는 끝에 greedy 정리 단계를 항상 돌린다.

## byte-diff 검증 결과

`run.sh` 가 자동 검증. 결과: **22/22 byte-identical**

- 본 실험 입력 3종 × 2모드 = 6 케이스: `output.*.mlir` == `intree.*.mlir`
- in-tree 회귀 테스트 8종 (`mlir/test/Dialect/Linalg/detensorize_*.mlir`)
  × 2모드 = 16 케이스: 전부 일치 (exit code 포함)

## 파일

```
input/while.mlir            0-d tensor while loop (CF model 발화 케이스)
input/pure-compute.mlir     제어흐름 무관 0-d generic (기본 비발화/aggressive 발화)
input/rank1-negative.mlir   rank-1 tensor (양 모드 비발화 negative)
output/output.<n>.mlir      my-detensorize 출력 (기본 모드)
output/intree.<n>.mlir      linalg-detensorize 출력 (기본 모드)
output/output.agg.<n>.mlir  my-detensorize{aggressive-mode} 출력
output/intree.agg.<n>.mlir  linalg-detensorize{aggressive-mode} 출력
run.sh                      전체 재현 + byte-diff (in-tree 테스트 16케이스 포함)
```

재현: `./run.sh` (전제: `out-of-tree/build/bin/my-mlir-opt` 빌드 완료)
