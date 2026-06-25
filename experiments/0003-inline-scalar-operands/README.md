# #0003 inline-scalar-operands — in-tree `linalg-inline-scalar-operands` 해부·재현

in-tree pass `linalg-inline-scalar-operands` 의 내부 구성(pass → pattern → 핵심 함수)을
해부하고, 같은 populate 함수 + 같은 driver 를 out-of-tree 에서 호출하는
`my-inline-scalar-operands` 로 재현해 byte-diff 로 검증한 실험.

## 호출 체인 (LLVM 19.1.7, 파일:라인)

```
LinalgInlineScalarOperandsPass                 InlineScalarOperands.cpp:103-115
  (def: Passes.td:85-90 — anchor 없는 Pass<"linalg-inline-scalar-operands">,
   옵션 0개, dependentDialects = ["linalg::LinalgDialect"])
  └─ runOnOperation()                          InlineScalarOperands.cpp:108-114
       ├─ populateInlineConstantOperandsPatterns(patterns)
       │       선언 Transforms.h:1722 / 정의 InlineScalarOperands.cpp:95-99
       │    └─ patterns.add<InlineScalarOperands>(ctx)        (pattern 1개)
       │         InlineScalarOperands : OpRewritePattern<GenericOp>   :34-90
       │         └─ matchAndRewrite(GenericOp, PatternRewriter&)      :36-89
       └─ applyPatternsAndFoldGreedily(op, std::move(patterns))       :113
```

## matchAndRewrite 내부 단계 ↔ IR 변화 매핑

| 코드 단계 (InlineScalarOperands.cpp) | IR 에 만드는 차이 |
|---|---|
| `:38-39` `hasPureTensorSemantics()` 아니면 bail | memref generic 은 불변 (negative (1)) |
| `:44-52` DPS input 중 `getMatchingIndexingMap(op).isConstant()` 인 것을 `scalarOperands` 로 수집, 나머지만 `newOperands`/`newIndexingMaps` 에 | `ins(...)` 목록과 `indexing_maps` 배열에서 scalar operand 제거 |
| `:54-55` scalar 없으면 bail | 전 input 이 비상수 map 이면 불변 (negative (2)) |
| `:57-59` DPS init 의 map 은 그대로 보존 | output 쪽 map 불변 — init 은 map 이 상수여도 후보 아님 |
| `:63-65` 줄어든 operand/map 으로 새 `GenericOp` 생성 | operand 수·map 수 감소한 새 generic |
| `:66-67` `cloneRegionBefore` 로 body 복제 | payload 연산은 그대로 유지 |
| `:73-85` 각 scalar 마다 (역순) body 선두에 `arith.constant index` + `tensor.extract` 생성 → 해당 block arg `replaceAllUsesWith` 후 `eraseArgument` | body 에 `%extracted = tensor.extract %t[...]` 등장, block arg 수 감소 (`^bb0(%in, %out)`) |
| `:87` `replaceOp` | 옛 generic 의 result 를 새 generic result 로 대체 |

'scalar' 판정 = **indexing map 의 모든 result 가 AffineConstantExpr**
(`AffineMap::isConstant()`, AffineMap.cpp:377-379 — `llvm::all_of(getResults(), IsaPred<AffineConstantExpr>)`).
rank-0 이냐 size-1 이냐가 아니라 *map 의 상수성*이 기준:

- (a) rank-0 tensor `tensor<f32>` — map `(d0) -> ()`, result 0개. empty range 의
  `all_of` 는 true → scalar 인정. `tensor.extract %t[]` (인덱스 0개) 로 inline.
- (b) 상수 인덱스 접근 — map `(d0,d1) -> (0,1)` 처럼 iteration 변수와 무관한
  고정 위치 읽기. `tensor.extract %t[%c0, %c1]` 로 inline.

## 입력/결과

| 입력 | 기대 | 실제 (output.* = intree.* byte-identical) |
|---|---|---|
| `input/rank0-scalar.mlir` | rank-0 input inline | ins 2→1, maps 3→2, body 에 `tensor.extract %arg0[]` 삽입, block arg 3→2 |
| `input/const-index.mlir` | `(0,1)` 고정 접근 inline | ins 2→1, `tensor.extract %arg0[%c0, %c1]`; `%c0/%c1` 은 **func entry 로 hoist** (아래 참고) |
| `input/negative.mlir` | 불변 | (1) memref semantics, (2) 비상수 map — 둘 다 generic 구조 불변 |

참고 — pattern 은 `arith.constant` 를 body 선두에 만들지만 (`:78-80`,
`rewriter.setInsertionPointToStart(body)` 후 create), 최종 IR 에서는
`applyPatternsAndFoldGreedily` 의 fold/constant 단계가 상수를 가장 가까운
isolated-from-above 영역(func) entry 로 끌어올린다. 즉 driver 까지가 한 세트로
최종 IR 모양을 만든다 — pattern 단독 산출물과 pass 산출물이 다를 수 있는 표본.

## 재현

```bash
./run.sh
# [OK ] byte-identical : const-index
# [OK ] byte-identical : negative
# [OK ] byte-identical : rank0-scalar
```

out-of-tree 재현 pass: `out-of-tree/lib/Passes/MyInlineScalarOperands.cpp`
(`my-inline-scalar-operands`). in-tree `runOnOperation()` 과 동일하게
`linalg::populateInlineConstantOperandsPatterns` + `applyPatternsAndFoldGreedily`
호출만 한다 (알고리즘 재구현 없음). 추가 link lib 불필요 — 기존
`MLIRLinalgTransforms`(populate/pattern 정의) + `MLIRTransforms`(greedy driver) 로 충분.

## 이식 메모 (개인 컴파일러 반영 시)

- 가져갈 것: `InlineScalarOperands` pattern 하나 (자기완결, ~60줄). 의존:
  `GenericOp` API (`getDpsInputOperands`/`getMatchingIndexingMap`/`getDpsInitsMutable`),
  `AffineMap::isConstant/getConstantResults`, `tensor::ExtractOp`, `arith::ConstantIndexOp`.
- pass 골격은 보일러플레이트: populate 한 줄 + greedy driver 한 줄.
- dependentDialects 는 linalg 만 선언해도 됨 — pattern 이 만드는 arith/tensor op 는
  LinalgDialect 의 dependent dialect 로 따라 로드된다.
