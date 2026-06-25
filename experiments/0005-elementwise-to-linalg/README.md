# #0005 elementwise-to-linalg — in-tree `convert-elementwise-to-linalg` 해부·재현

in-tree pass `convert-elementwise-to-linalg` 의 내부 구성(pass → driver → pattern →
핵심 함수)을 해부하고, 같은 populate 함수 + 같은 driver 를 out-of-tree 에서 호출하는
`my-elementwise-to-linalg` 로 재현해 byte-diff 로 검증한 실험.

기능: **ElementwiseMappable trait 을 가진 op (arith.addf, math.exp, arith.cmpf 등) 이
ranked tensor operand 위에서 동작할 때 → `linalg.generic` (identity map × all-parallel)
으로 치환**. op 종류 목록이 아니라 *trait 기반* — dialect 를 가리지 않는다.

본 cycle 처음으로 driver 가 greedy 가 아니라 **dialect-conversion**
(`ConversionTarget` + `applyPartialConversion`) 인 pass.

## 호출 체인 (LLVM 19.1.7, 파일:라인)

```
ConvertElementwiseToLinalgPass                 ElementwiseToLinalg.cpp:122-142
  (def: Passes.td:14-25 — Pass<"convert-elementwise-to-linalg", "">,
   anchor "" = op-agnostic, 옵션 0개,
   dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"])
  └─ runOnOperation()                          ElementwiseToLinalg.cpp:128-141
       ├─ ConversionTarget target(*context)                          :131
       ├─ populateElementwiseToLinalgConversionPatterns(patterns)    :134
       │       선언 Transforms.h:1642 / 정의 ElementwiseToLinalg.cpp:115-119
       │    └─ patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(ctx)
       │         RewritePattern(MatchAnyOpTypeTag(), benefit=1)      :76-78
       │           — 특정 op type 이 아니라 모든 op 에 매치 시도하는 pattern
       │         └─ matchAndRewrite(Operation *, PatternRewriter &)  :79-111
       │              ├─ isElementwiseMappableOpOnRankedTensors(op) 아니면
       │              │  notifyMatchFailure                          :81-83
       │              ├─ indexingMaps = (numResults+numOperands) 개의
       │              │  rewriter.getMultiDimIdentityMap(rank)       :85-88
       │              ├─ iteratorTypes = rank × parallel             :89-90
       │              ├─ outputs = getOrCreateOperandsMatchingResultTypes
       │              │  (rewriter, op)            (정의 :46-73)     :91
       │              └─ rewriter.replaceOpWithNewOp<linalg::GenericOp>
       │                 (..., bodyBuilder)                          :92-109
       │                   bodyBuilder: 원래 op 을 scalar 버전으로 재생성
       │                   builder.create(loc, op->getName().getIdentifier(),
       │                     regionArgs.take_front(numOperands),
       │                     resultTypes(elemental), op->getAttrs()) :104-107
       │                   + linalg::YieldOp(scalarOp->getResults()) :108
       ├─ target.markUnknownOpDynamicallyLegal([](Operation *op) {
       │       return !isElementwiseMappableOpOnRankedTensors(op);
       │  })                                                         :135-137
       └─ applyPartialConversion(func, target, std::move(patterns))  :139
            → dialect-conversion driver (greedy 아님). 실패 시
              signalPassFailure()                                    :139-140
```

legality 술어 `isElementwiseMappableOpOnRankedTensors` (ElementwiseToLinalg.cpp:24-31,
file-static):

```cpp
OpTrait::hasElementwiseMappableTraits(op)   // 선언 OpDefinition.h:1502,
                                            // 정의 Operation.cpp:1393-1396 —
                                            // Elementwise && Scalarizable &&
                                            // Vectorizable && Tensorizable 전부
&& llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>)
```

같은 술어가 (a) pattern 의 match 전제 (:81) 와 (b) ConversionTarget 의 legality
(:135, **부정형**) 양쪽에 쓰인다 — "illegal = 변환 대상 = pattern 적용 가능" 이
정확히 일치하므로 partial conversion 이 항상 성공한다 (잔존 illegal 불가능).

## 코드 단계 ↔ IR 변화 매핑

| 코드 단계 (ElementwiseToLinalg.cpp) | IR 에 만드는 차이 |
|---|---|
| `:24-31` 술어 — 4 trait + all operand ranked tensor | scalar(f32)·vector(vector<4xf32>)·scalar 가 섞인 select (negative 3종) 모두 불변. `:28-29` TODO 가 말하듯 all_of 라서 scalar-tensor 혼합 broadcast 는 미지원 |
| `:85-88` indexingMaps | 모든 ins/outs 에 동일한 `#map = affine_map<(d0,d1) -> (d0,d1)>` (identity) — rank 는 result 0 기준 |
| `:89-90` iteratorTypes | `["parallel", "parallel"]` — 전부 parallel (elementwise 니까) |
| `:46-73` outputs 채우기 (found 분기 :57-63) | result type == 어떤 operand type 이면 그 operand 를 DPS init 으로 재사용 → `outs(%arg0 ...)` 처럼 입력이 outs 에 그대로 등장 (값은 안 읽음 — body 가 %out 을 사용하지 않는 pure init) |
| `:46-73` outputs 채우기 (empty 분기 :67-70) | result type 이 모든 operand 와 다르면 (`arith.cmpf` 의 i1 결과) `tensor.empty()` 생성. dynamic shape 면 `tensor::getMixedSizes` 가 **첫 operand** 에서 dim 마다 `arith.constant index` + `tensor.dim` 을 materialize → `tensor.empty(%dim, %dim_0)` |
| `:92-109` GenericOp 으로 치환 + bodyBuilder :99-108 | tensor op 1개 → `linalg.generic` 1개. body 에 **같은 이름의 scalar op** 재생성 (`arith.addf %in, %in_0 : f32`). attr 도 `op->getAttrs()` 로 통과 — `cmpf ogt` 의 predicate 가 body 의 scalar cmpf 에 보존 |

발화 조건 요약 — **(a) Elementwise+Scalarizable+Vectorizable+Tensorizable 4 trait
전부 보유 (= `ElementwiseMappable.traits`), (b) 모든 operand 가 ranked tensor**.
결과는 op 당 `linalg.generic` 1개 (+ 필요 시 `tensor.empty`/`tensor.dim`).

## 입력/결과

| 입력 | 기대 | 실제 (output.* = intree.* byte-identical) |
|---|---|---|
| `input/static.mlir` | addf/mulf/exp/cmpf/select 5개 발화 | generic 5개. addf/mulf/exp 는 operand 재사용 (`outs(%arg0)`, `outs(%0)`, `outs(%1)`), cmpf 는 `tensor.empty() : tensor<8x16xi1>` 신규, select 는 indexing map 4개 (i1 tensor 포함 3 ins + 1 out) |
| `input/dynamic.mlir` | addf 는 재사용, cmpf 는 dim 추출 | cmpf 앞에 `arith.constant 0/1 : index` + `tensor.dim %0, %c0/%c1` ×2 + `tensor.empty(%dim, %dim_0)` — `getMixedSizes` 가 첫 operand(%0) 에서 추출 |
| `input/negative.mlir` | 불변 | (1) scalar addf (2) vector addf (3) scalar-cond select — 모두 불변 (all_of 가드) |

## 재현

```bash
./run.sh
# [OK ] byte-identical : dynamic
# [OK ] byte-identical : negative
# [OK ] byte-identical : static
```

out-of-tree 재현 pass: `out-of-tree/lib/Passes/MyElementwiseToLinalg.cpp`
(`my-elementwise-to-linalg`). in-tree `runOnOperation()` 과 동일하게
`linalg::populateElementwiseToLinalgConversionPatterns` + `ConversionTarget`
(`markUnknownOpDynamicallyLegal`) + `applyPartialConversion` 을 호출 (알고리즘
재구현 없음). 단 legality 술어는 in-tree 에서 file-static (:24-31) 이라 export
되지 않으므로 같은 2줄을 exported building block
(`OpTrait::hasElementwiseMappableTraits`, MLIRIR) 으로 구성. link lib 는 기존
`MLIRLinalgTransforms`(populate/pattern) + `MLIRTransforms`(driver) 에
`MLIRMemRefDialect` 추가 (in-tree Passes.td:24 의 dependentDialects 그대로
`memref::MemRefDialect` 직접 참조).

## 이식 메모 (개인 컴파일러 반영 시)

- 가져갈 것: pattern 1개 (`ConvertAnyElementwiseMappableOpOnRankedTensors`,
  ~35줄) + 헬퍼 2개 (`isElementwiseMappableOpOnRankedTensors` 2줄,
  `getOrCreateOperandsMatchingResultTypes` ~27줄). 의존:
  `OpTrait::hasElementwiseMappableTraits` (MLIRIR), `linalg::GenericOp`
  bodyBuilder builder, `tensor::EmptyOp` + `tensor::getMixedSizes`.
- driver 는 greedy 로 바꿔도 동작은 같을 것이나 (pattern 이 단조 — generic 은
  ElementwiseMappable 이 아니라 재발화 없음), in-tree 선택은 dialect-conversion:
  "elementwise-on-ranked-tensor 가 0개" 라는 *종료 상태를 선언* 하고 미달 시
  pass 실패로 알려주는 안전망이 공짜로 따라온다.
- 술어와 legality 가 정확히 부정 관계인 구성이 핵심 — pattern 의 match 전제를
  바꾸면 target 의 legality 도 같이 바꿔야 한다 (아니면 잔존 illegal 로 pass
  실패).
- scalar 가 섞인 broadcast (`select %scalar_cond, %t, %t`) 는 미지원 (:28-29
  TODO). 이식 후 확장하려면 scalar operand 를 추적해 indexing map 을
  `() -> ()` 형으로 달리 주는 일반화가 필요.
