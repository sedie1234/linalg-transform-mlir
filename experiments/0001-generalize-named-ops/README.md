# 0001 — linalg-generalize-named-ops 해부·재현·관찰

in-tree pass `linalg-generalize-named-ops` (named op → `linalg.generic`)를
해부(A)하고, out-of-tree pass `my-generalize-named-ops`로 재현(B)하고,
IR 전후 변화를 관찰(C)한 실험. 상세 로그: `../../log/0001-generalize-named-ops.html`.

## 재현

```bash
./run.sh
# 각 input/*.mlir 에 대해 output/output.<name>.mlir (my pass) 와
# output/intree.<name>.mlir (in-tree pass) 를 캡처하고 byte-diff.
# 결과: 3/3 byte-identical (2026-06-12, LLVM 19.1.7)
```

## A. 호출 체인 (파일:라인 — LLVM 19.1.7)

```
Passes.td:92-95          def LinalgGeneralizeNamedOpsPass : Pass<"linalg-generalize-named-ops">
                         (옵션 0개, dependentDialects = ["linalg::LinalgDialect"], anchor 없음)
Generalization.cpp:79-85 struct LinalgGeneralizeNamedOpsPass : impl::...PassBase<...>
Generalization.cpp:89-93 runOnOperation()
  ├─ :91  populateLinalgNamedOpsGeneralizationPatterns(patterns)
  │         정의 Generalization.cpp:95-98 / 선언 Transforms.h:1588
  │         └─ patterns.add<LinalgGeneralizationPattern>(ctx)
  │              Transforms.h:1408-1421  (OpInterfaceRewritePattern<LinalgOp>)
  │              └─ matchAndRewrite → returningMatchAndRewrite
  │                   └─ generalizeNamedOp(rewriter, op)
  │                        선언 Transforms.h:693 / 정의 Generalization.cpp:53-75
  │                        ├─ generalizeNamedOpPrecondition  :38-51
  │                        ├─ getDpsInputs/getDpsInits/getIndexingMapsArray/
  │                        │  getIteratorTypesArray           :58-61
  │                        ├─ resultTypes 분기 (tensor vs memref) :62-64
  │                        ├─ rewriter.create<GenericOp>(...)  :69-70
  │                        ├─ rewriter.inlineRegionBefore(...) :71-72
  │                        └─ rewriter.replaceOp(...)          :73
  └─ :92  applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
          ← driver = greedy
```

## B. out-of-tree 재현 (4-edit)

| 파일 | 변경 |
|------|------|
| `out-of-tree/lib/Passes/MyGeneralizeNamedOps.cpp` | 신규 — in-tree runOnOperation 과 동일 3줄 (populate + greedy) |
| `out-of-tree/lib/Passes/CMakeLists.txt` | SOURCES 에 한 줄 추가 |
| `out-of-tree/lib/Passes/PassRegistration.cpp` | `registerPass(createMyGeneralizeNamedOpsPass)` |
| `out-of-tree/include/MyPasses/Passes.h` | factory 선언 + 호출 체인 요약 주석 |

link libs 는 기존 `MLIRLinalgTransforms`(populate/generalizeNamedOp 정의) +
`MLIRTransforms`(greedy driver)로 충분 — 추가 없음.

## C. 코드 단계 ↔ IR 변화 매핑

| 입력 | 발화 op | 코드 단계 (Generalization.cpp) | IR 변화 |
|------|---------|-------------------------------|---------|
| `tensor-named-ops.mlir` | `linalg.matmul` | `getIndexingMapsArray` :60 — named op 가 내장한 map 을 그대로 추출 | `#map (d0,d1,d2)->(d0,d2) / (d2,d1) / (d0,d1)` 가 `indexing_maps` 속성으로 노출 |
| 〃 | 〃 | `getIteratorTypesArray` :61 | `iterator_types = ["parallel","parallel","reduction"]` 노출 |
| 〃 | 〃 | `inlineRegionBefore` :71-72 — region 재생성이 아니라 **이동** | 숨겨져 있던 payload `arith.mulf + arith.addf + linalg.yield` 가 명시적 `^bb0` block 으로 |
| 〃 | `linalg.add` | 같은 pattern 1개가 인터페이스 match | elementwise generic (`#map3` identity ×3) |
| 〃 | `linalg.transpose` | permutation 속성이 이미 indexing map 으로 내장돼 있음 | `#map4 (d0,d1)->(d1,d0)` 입력 map, payload 는 `yield` 단독 |
| `memref-named-ops.mlir` | `linalg.fill`, `linalg.matmul` | `hasPureTensorSemantics()==false` → `resultTypes=TypeRange{}` :62-64 | 결과값 없는 `linalg.generic` (메모리 부수효과만). fill 의 scalar map 은 `(d0,d1)->()` |
| `negative-generic-map.mlir` | (발화 없음) | `generalizeNamedOpPrecondition` :42 — `isa<GenericOp> || isa<MapOp>` bail | `linalg.generic`/`linalg.map` 그대로 (재인쇄만) |

핵심 통찰: **named op 는 이미 generic 의 모든 정보(indexing_maps, iterator_types,
payload region)를 내부에 갖고 있다.** generalization 은 정보를 *계산*하는 게 아니라
숨겨진 표현을 *노출*만 한다 — region 조차 `inlineRegionBefore` 로 이동할 뿐.

## byte-diff 결과

```
[OK ] byte-identical : memref-named-ops
[OK ] byte-identical : negative-generic-map
[OK ] byte-identical : tensor-named-ops
```
