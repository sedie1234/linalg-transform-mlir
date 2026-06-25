# #0006 fuse-elementwise-ops — in-tree `linalg-fuse-elementwise-ops` 해부·재현

in-tree pass `linalg-fuse-elementwise-ops` 의 내부 구성(pass → driver → pattern →
핵심 함수)을 해부하고, 같은 populate 함수들 + 같은 greedy driver 를 out-of-tree 에서
호출하는 `my-fuse-elementwise-ops` 로 재현해 byte-diff 로 검증한 실험.

기능: **elementwise `linalg.generic` producer-consumer 쌍을 indexing map 합성으로
한 generic 으로 병합** (+ reshape-by-expansion 전파, fill/splat/outs fold,
canonicalization, constant fold 를 greedy 고정점까지). 본 cycle 첫 *대형 합주* pass —
fusion 본체 1개가 아니라 11+개 패턴 묶음이며, in-tree 주석(:2122-2127)이 명시하듯
"테스트용, deprecate 예정" — 실전(downstream)은 populate 함수를 자체 cost model
(`ControlFusionFn`)과 직접 조합한다.

## 호출 체인 (LLVM 19.1.7, 파일:라인)

```
LinalgElementwiseOpFusionPass                  ElementwiseOpFusion.cpp:2128-2164
  (def: Passes.td:73-78 — Pass<"linalg-fuse-elementwise-ops">,
   anchor 없음 = op-agnostic, 옵션 0개,
   dependentDialects = ["affine::AffineDialect", "linalg::LinalgDialect",
                        "memref::MemRefDialect"])
  └─ runOnOperation()                          ElementwiseOpFusion.cpp:2133-2163
       ├─ defaultControlFn = [](OpOperand *fusedOperand) {
       │      producer = fusedOperand->get().getDefiningOp();
       │      return producer && producer->hasOneUse(); }          :2139-2142
       │    — ControlFusionFn (= std::function<bool(OpOperand*)>,
       │      Transforms.h:1650). 합법성과 별개의 "할지 말지" 비용 훅.
       │      모든 populate 호출에 이 동일 controlFn 이 들어간다.
       ├─ populateElementwiseOpsFusionPatterns(patterns, ctrl)      :2145
       │    선언 Transforms.h:1656 / 정의 ElementwiseOpFusion.cpp:2097-2106
       │    ├─ FuseElementwiseOps                          :417-456  ★fusion 본체
       │    │    matchAndRewrite(:424-452): consumer 의 각 OpOperand 에
       │    │      ├─ areElementwiseOpsFusable(&opOperand)          :93-167
       │    │      │    (선언 Transforms.h:452 — 합법성 술어, 아래 표)
       │    │      ├─ controlFn(&opOperand)                         :430
       │    │      └─ fuseElementwiseOps(rewriter, &opOperand)      :292-413
       │    │           (선언 Transforms.h:503-504)
       │    │           ├─ getPreservedProducerResults              :76-90
       │    │           │    (consumer 밖에서도 쓰이는 producer 결과 보존)
       │    │           ├─ operand·indexing map 병합 :308-370 — producer 입력
       │    │           │   map 은 getIndexingMapOfProducerOperandsIn
       │    │           │   CoordinatesOfFusedOp(:44-71):
       │    │           │   argMap ∘ inv(producerResultMap) ∘ consumerArgMap
       │    │           ├─ rewriter.create<GenericOp>(병합 결과)    :373-378
       │    │           │   (iterator_types 는 consumer 것 그대로 :376)
       │    │           └─ generateFusedElementwiseOpRegion         :171-290
       │    │               (두 payload block 을 한 block 으로 splice;
       │    │                linalg.index 는 consumerToProducerLoopsMap
       │    │                (:398-399 합성) 으로 remap)
       │    ├─ FoldFillWithGenericOp                       :2047-2074
       │    │    input 을 정의한 linalg.fill 의 스칼라를 payload 에 직결
       │    ├─ FoldScalarOrSplatConstant                   :1893-1993
       │    │    splat/scalar 상수 input 을 body 안 스칼라 상수로 + operand 제거
       │    ├─ RemoveOutsDependency                        :2006-2044
       │    │    payload 가 안 읽는 outs 를 tensor.empty 로 교체 (fusion 기회 확대)
       │    └─ populateEraseUnusedOperandsAndResultsPatterns
       │         (선언 Transforms.h:1671 / 정의
       │          EraseUnusedOperandsAndResults.cpp:421) — dead 청소
       ├─ populateFoldReshapeOpsByExpansionPatterns(patterns, ctrl) :2146
       │    선언 Transforms.h:1692 / 정의 ElementwiseOpFusion.cpp:2077-2086
       │    ├─ FoldReshapeWithGenericOpByExpansion         :1024-1088
       │    │    (generic 결과를 먹는 tensor.expand_shape 흡수)
       │    ├─ FoldPadWithProducerReshapeOpByExpansion     :959-1020
       │    │    (collapse_shape→pad 를 pad→collapse_shape 로 교환)
       │    └─ FoldWithProducerReshapeOpByExpansion        :922-957
       │         (input 의 tensor.collapse_shape 흡수)
       │         — 셋의 공통 엔진 fuseWithReshapeByExpansion(:775-920),
       │           ExpansionInfo(:544-623) 가 reassociation 으로 루프 차원 확장
       ├─ canonicalization 5그룹                            :2149-2154
       │    AffineApplyOp / GenericOp / tensor::ExpandShapeOp /
       │    tensor::CollapseShapeOp::getCanonicalizationPatterns +
       │    LinalgDialect->getCanonicalizationPatterns
       ├─ populateConstantFoldLinalgOperations(patterns, ctrl)      :2157
       │    선언 Transforms.h:1701 / 정의 ConstantFold.cpp:306
       └─ applyPatternsAndFoldGreedily(op, patterns, grc)     :2160-2162
            → greedy driver, GreedyRewriteConfig
              useTopDownTraversal = true (:2159 — 컴파일 시간 사유)
```

주의: 같은 파일의 **collapse 계열** (`populateFoldReshapeOpsByCollapsingPatterns`
:2088-2095, `populateCollapseDimensions` :2108-2114) 은 이 pass 의 runOnOperation
에 **포함되지 않는다** — expansion 방향만 쓴다.

## 합법성 술어 `areElementwiseOpsFusable` (:93-167)

| 검사 (라인) | 내용 |
|---|---|
| :97-102 | producer·consumer 둘 다 `linalg.generic` |
| :107-109 | producer 는 pure tensor semantics + fusedOperand 가 RankedTensorType |
| :113-114 | producer 의 모든 iterator 가 parallel (reduction producer 탈락) |
| :118-119 | fusedOperand 는 consumer 의 **DPS input** (output fusion 은 TODO) |
| :123-125 | consumer arg map 결과수 == producer 루프수 |
| :129-132 | producer result map 이 **permutation** (역함수 합성에 필요) |
| :138-164 | consumer 에 reduction 이 있으면 fusion 후 모든 루프 dim 이 어느 입력 map 에든 등장하는지 (loop bound 정보 보존) |

## 코드 단계 ↔ IR 변화 매핑

| 입력 (input/) | 발화 pattern → 핵심 함수 | IR 변화 (output/ 의 before→after) |
|---|---|---|
| `chain.mlir` @add_mul_sub_chain | `FuseElementwiseOps`(:417) → `fuseElementwiseOps`(:292) ×2회 (greedy 고정점) | generic 3개(add→mul→sub 체인) → **generic 1개**. body 에 `arith.addf`+`mulf`+`subf` 가 한 block 으로 splice(:171-290). ins 가 (%a,%b)+(%c)+(%d) 로 병합돼 4개, indexing map 5개 전부 identity(#map). 중간 결과 텐서·dead `tensor.empty` 2개는 greedy DCE 로 소멸 |
| `reshape-splat.mlir` @collapse_into_generic | `FoldWithProducerReshapeOpByExpansion`(:922) → `fuseWithReshapeByExpansion`(:775) | input 의 `tensor.collapse_shape`(3D→2D) 가 generic 안으로 흡수: generic 이 2D→**3D** (iterator 3개, map 이 (d0,d1,d2)) 로 확장되고, 다른 operand %b 에 `tensor.expand_shape` 가, 결과에 `tensor.collapse_shape` 가 생성 — reshape 가 generic *아래로* 전파 |
| `reshape-splat.mlir` @splat_fold | `FoldScalarOrSplatConstant`(:1893) | `arith.constant dense<2.0> : tensor<4x8xf32>` (splat) input 이 operand 에서 제거되고 body 안 `arith.constant 2.0 : f32` 스칼라로 (:1965-1966). ins 2→1, indexing map 3→2 |
| `negative.mlir` @multi_use_producer | (발화 없음) — `controlFn`(:430) 이 차단: producer %add 가 2 uses → `hasOneUse()`(:2141) false | 출력 == round-trip 입력. 합법성은 통과하지만 *비용 훅* 이 거부하는 케이스 |
| `negative.mlir` @reduction_then_elementwise | (발화 없음) — `areElementwiseOpsFusable`(:113-114) 차단: producer 가 reduction iterator 보유 | 출력 == round-trip 입력. row-sum generic 과 후속 elementwise generic 이 그대로 분리 유지 |

## out-of-tree 재현 (`my-fuse-elementwise-ops`)

- 코드: `out-of-tree/lib/Passes/MyFuseElementwiseOps.cpp`
- in-tree `runOnOperation()`(:2133-2163) 과 **같은 populate 4종 + canonicalization
  5그룹 + 같은 defaultControlFn + 같은 greedy(top-down) driver** 를 같은 순서로 호출.
  알고리즘 재구현 없음.
- link lib 추가: `MLIRAffineDialect` (dependentDialects 의 affine +
  `AffineApplyOp::getCanonicalizationPatterns` 참조).

## 재현 방법

```bash
./run.sh
# 각 입력 i 에 대해
#   output/output.<i>.mlir  ← my-mlir-opt --my-fuse-elementwise-ops
#   output/intree.<i>.mlir  ← my-mlir-opt --linalg-fuse-elementwise-ops
# 를 캡처하고 diff. 2026-06-12 결과: 3/3 byte-identical.
```

## affine 대비 (학습 baseline)

affine #0023 의 ad-hoc loop fusion 이 *의존성 분석* (메모리 접근 교차 검사) 으로
합법성을 증명하는 것과 달리, linalg elementwise fusion 은 **indexing map 의 대수적
합성** (`argMap ∘ inv(resultMap) ∘ consumerMap`) 하나로 합법성(permutation 검사)과
결과 map 계산을 동시에 해결한다 — SSA + 구조적 op 의 이점.
