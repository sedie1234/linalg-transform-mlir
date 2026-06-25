# #0007 fold-unit-extent-dims — in-tree `linalg-fold-unit-extent-dims` 해부·재현

in-tree pass `linalg-fold-unit-extent-dims` 의 내부 구성(pass → driver → pattern →
핵심 함수)을 해부하고, 같은 populate 함수들 + 같은 greedy driver 를 out-of-tree 에서
호출하는 `my-fold-unit-extent-dims` 로 재현해 byte-diff 로 검증한 실험.

기능: **broadcasting 표현용 unit-extent dimension (size 1) 을 linalg op 의
operand/iteration space 에서 제거**. unit dim 판정은 indexing map 의 역사상
(`inversePermutation(concatAffineMaps(maps))`) 으로 하고, 제거는 dim 의
AffineConstantExpr(0) 치환 + operand collapse / 결과 expand 로 한다.
옵션 `use-rank-reducing-slices` 가 collapse/expand 의 *op 종류* 를 가른다:
기본 `ReassociativeReshape` = `tensor.collapse_shape`/`expand_shape`
(메타데이터 변환), 옵션 시 `ExtractInsertSlice` = `tensor.extract_slice`/
`insert_slice` (복사 의미론, bufferization 친화성이 다름).

## 호출 체인 (LLVM 19.1.7, 파일:라인)

```
LinalgFoldUnitExtentDimsPass                       DropUnitDims.cpp:817-835
  (def: Passes.td:61-71 — Pass<"linalg-fold-unit-extent-dims", "">,
   anchor "" = op-agnostic. 옵션 1개: useRankReducingSlices /
   "use-rank-reducing-slices" / bool / default false.
   dependentDialects = ["linalg::LinalgDialect", "affine::AffineDialect",
                        "memref::MemRefDialect"])
  └─ runOnOperation()                              DropUnitDims.cpp:822-834
       ├─ ControlDropUnitDims options              :826 (Transforms.h:473-489)
       │    · rankReductionStrategy 기본 = ReassociativeReshape (Th:476-477)
       │    · controlFn 기본 (Th:479-488) = GenericOp 모든 loop dim /
       │      tensor.PadOp 모든 source dim 허용
       ├─ if (useRankReducingSlices)
       │    options.rankReductionStrategy = ExtractInsertSlice   :827-830
       │    ★ 옵션이 가르는 유일한 분기 — populate 선택 + collapse/expand
       │      op 종류가 전부 여기서 갈린다.
       ├─ populateFoldUnitExtentDimsPatterns(patterns, options)  :831
       │    선언 Transforms.h:1715-1716 / 정의 DropUnitDims.cpp:798-808
       │    strategy 2-way 분기:
       │    ┌─ ReassociativeReshape →
       │    │  populateFoldUnitExtentDimsViaReshapesPatterns     :762-780
       │    │    ├─ DropUnitDims                     :553-566  ★본체
       │    │    │    matchAndRewrite(:558-561) → linalg::dropUnitDims
       │    │    │      (정의 :389-550 / 선언 Transforms.h:491-492)
       │    │    │      1. inversePermutation(concatAffineMaps(maps)) 로
       │    │    │         iteration dim→operand dim 역사상; static size==1
       │    │    │         ∧ controlFn 허용 → unitDims        :395-420
       │    │    │      2. unit dim → AffineConstantExpr(0) 치환,
       │    │    │         나머지 재번호 → newIteratorTypes    :422-440
       │    │    │      3. operand 별 dropUnitExtentFromOperandMetadata
       │    │    │         (:340-387) — 새 map·targetShape·reassociation.
       │    │    │         hasCollapsibleType(:457-466): identity layout
       │    │    │         memref / encoding 없는 tensor 만.
       │    │    │         abort(:489-493): newIndexingMaps==indexingMaps
       │    │    │         또는 역사상 불가 → failure (negative 경로)
       │    │    │      4. collapseValue(:284-328) — Reshape 전략:
       │    │    │         tensor/memref.collapse_shape; Slice 전략:
       │    │    │         tensor.ExtractSliceOp / memref.SubViewOp 의
       │    │    │         rankReduceIfNeeded
       │    │    │      5. 새 GenericOp + region inline        :512-526
       │    │    │      5a. replaceUnitDimIndexOps(:232-251) — dropped
       │    │    │          dim 의 linalg.index → arith.constant 0,
       │    │    │          나머지는 index 번호 시프트
       │    │    │      6. expandValue(:256-279) — 결과를 원형 type 으로
       │    │    │         (expand_shape 또는 insert_slice) → replaceOp
       │    │    ├─ DropPadUnitDims                  :573-685
       │    │    │    tensor.pad 의 size==1 ∧ low/high==0 dim 을 collapse
       │    │    │    하고 낮은 rank 로 pad 후 expand (Slice 모드면
       │    │    │    tensor.empty + insert_slice 로 복원 :658-674)
       │    │    ├─ RankReducedExtractSliceOp        :690-720
       │    │    ├─ RankReducedInsertSliceOp<InsertSliceOp>
       │    │    │  RankReducedInsertSliceOp<ParallelInsertSliceOp> :724-757
       │    │    └─ 보조: FillOp/CollapseShapeOp/EmptyOp/ExpandShapeOp
       │    │       canonicalization (:773-776),
       │    │       tensor::populateFoldTensorEmptyPatterns (:777),
       │    │       memref::populateResolve{Ranked,}ShapedTypeResultDims
       │    │       Patterns (:778-779)
       │    └─ ExtractInsertSlice →
       │       populateFoldUnitExtentDimsViaSlicesPatterns      :782-796
       │         options.rankReductionStrategy 강제 재설정(:786-787) 후
       │         DropUnitDims + DropPadUnitDims 만.
       │         RankReduced*SliceOp 3종 + CollapseShape/ExpandShape
       │         canonicalization 은 **추가하지 않음** (:788-795)
       ├─ populateMoveInitOperandsToInputPattern(patterns)       :832
       │    선언 Transforms.h:1719 / 정의 DropUnitDims.cpp:810-813
       │    └─ MoveInitOperandsToInput              :83-163
       │         pure-tensor ∧ all-parallel generic 에서 body 가 읽는
       │         init(outs) 을 ins 로 옮기고 outs 는 새 tensor.empty 로 —
       │         unit reduction dim 이 모두 fold 된 op 의 후처리
       └─ applyPatternsAndFoldGreedily(op, std::move(patterns))  :833
            → **greedy driver** (GreedyRewriteConfig 기본값)
```

주의: 같은 파일 끝의 `populateContractionOpRankReducingPatterns`
(:1067-1097 — `RankReduceMatmul`/`RankReduceToUnBatched`, named contraction
op 의 unit dim 강하. 예: matmul→vecmat, batch_matmul→matmul) 는 이 pass 의
runOnOperation 에 **포함되지 않는다** (transform dialect 등 다른 경로용).

## 재현 (out-of-tree)

- pass: `out-of-tree/lib/Passes/MyFoldUnitExtentDims.cpp` → `my-fold-unit-extent-dims`
- in-tree 가 export 하는 `linalg::populateFoldUnitExtentDimsPatterns` +
  `linalg::populateMoveInitOperandsToInputPattern` (둘 다 lib `MLIRLinalgTransforms`)
  을 가져와 in-tree runOnOperation(:822-834) 과 동일 절차·동일 순서로 호출.
  알고리즘 재구현 없음.
- 옵션 `use-rank-reducing-slices` 도 같은 이름·같은 기본값(false)으로
  `Pass::Option<bool>` 노출 (PassWrapper 에는 copy ctor 필요 — clonePass 가
  copyOptionValuesFrom 으로 옵션 값을 복사).
- 추가 링크 불필요 (기존 `MLIRLinalgTransforms` + `MLIRTransforms` 로 충분).

## 실행

```bash
./run.sh
```

각 입력 × 2 모드(기본 reshape / `use-rank-reducing-slices=true`)에 대해
`--my-fold-unit-extent-dims` vs `--linalg-fold-unit-extent-dims` 출력을
byte-diff. **결과: 8/8 byte-identical** (2026-06-12, LLVM 19.1.7).

## 코드 단계 ↔ IR 변화 매핑

### input/broadcast.mlir — `@broadcast_add` (positive: DropUnitDims, operand만)

| 코드 단계 (DropUnitDims.cpp) | IR 변화 |
|---|---|
| dropUnitDims 1단계(:395-420): iteration dim 은 d0,d1 둘 다 5 → `unitDims` 공집합 | iterator_types 2개 유지 |
| 3단계 dropUnitExtentFromOperandMetadata(:340-387): `(0,d1)` 의 상수0 자리·`(d0,0)` 의 상수0 자리가 isUnitDim(:351-364) 의 *AffineConstantExpr(0) ∧ size==1* 분기로 fold | `#map0 (d0,d1)->(0,d1)` → `(d0,d1)->(d1)`, `#map1 (d0,d1)->(d0,0)` → `(d0,d1)->(d0)` |
| 4단계 collapseValue(:284-328, Reshape 전략 :308-326) | `tensor.collapse_shape %arg0 [[0,1]] : tensor<1x5xf32> into tensor<5xf32>` ×2 생성 |
| 6단계: 출력 operand 는 unit dim 없음 → collapsed[2]=false, expandValue 불호출 | 결과 type `tensor<5x5xf32>` 그대로, expand_shape 없음 |

→ "iteration dim 이 하나도 안 떨어져도" 상수-0 접근 정리만으로
`newIndexingMaps != indexingMaps` 가 되어 rewrite 되는 경로 (:448-452 주석의
legacy 동작) 를 입증.

### input/broadcast.mlir — `@drop_unit_loop_with_index` (positive: unit loop + linalg.index)

| 코드 단계 | IR 변화 |
|---|---|
| 1단계: d0 가 `tensor<1x8>` 의 size-1 자리에 닿음 → unitDims={0} | `iterator_types ["parallel","parallel"]` → `["parallel"]` |
| 2단계(:422-440): d0→const0, d1→d0 재번호 | `#map (d0,d1)->(d0,d1)` → `(d0)->(d0)` |
| 5a replaceUnitDimIndexOps(:232-251): `linalg.index 0`(dropped) → `arith.constant 0` (이후 greedy fold 로 `addi 0,j → j` 소멸), `linalg.index 1` → `linalg.index 0` 시프트 | body 에 `linalg.index 0` 1개만 남음 |
| 6단계 expandValue(:256-279) | `tensor.expand_shape %1 [[0,1]] output_shape [1,8]` 로 원형 복원 |
| 보조 populateFoldTensorEmptyPatterns(:777) | `tensor.empty()` 가 `tensor<8xf32>` 로 직접 재생성 (empty+collapse fold) |

### input/unit-reduction.mlir — `@unit_reduction` (positive: DropUnitDims + MoveInitOperandsToInput 합주)

| 코드 단계 | IR 변화 |
|---|---|
| 1단계: d0(par,1)·d2(red,1)·d3(red,1) 이 one-trip → unitDims={0,2,3} | 4-loop → 1-loop, `["parallel","parallel","reduction","reduction"]` → `["parallel"]` |
| 3·4단계 | `tensor.collapse_shape %arg0 [[0,1,2,3]] : tensor<1x?x1x1xf32> into tensor<?xf32>`; init `tensor<1x1xf32>` → `tensor<f32>` (rank-0) |
| FillOp canonicalization(:773) + CollapseShape canonicalization | `linalg.fill ... outs(tensor<f32>)` 로 fill 자체가 rank-0 으로 강하 |
| MoveInitOperandsToInput(:83-163): all-parallel 이 된 후 body 가 `%out` 을 읽으므로 candidates 비공집합 → init 을 ins 로 | `ins(%collapsed, %1 : tensor<?xf32>, tensor<f32>)` + `outs(%2 = tensor.empty() : tensor<f32>)` — fill 결과가 **입력**이 됨 |
| 6단계 expandValue | `tensor.expand_shape %3 [] output_shape [1,1] : tensor<f32> into tensor<1x1xf32>` |

→ reduction 이었던 op 가 *elementwise generic* 으로 변해 fusion 대상이 되는,
DropUnitDims.cpp:47-82 주석 시나리오 그대로.

### input/pad-slice.mlir (positive: 비-generic 패턴 3종 + 모드 분기)

| 함수 | reshape 모드 (기본) | slices 모드 (`use-rank-reducing-slices=true`) |
|---|---|---|
| `@pad_unit` | DropPadUnitDims(:573-685): `collapse_shape` → 1-D `tensor.pad low[2] high[2]` → `expand_shape` | 같은 패턴, collapseValue/expandValue 가 Slice 분기: `extract_slice` → 1-D pad → `tensor.empty()+insert_slice` (:658-674) |
| `@slice_unit` | RankReducedExtractSliceOp(:690-720): rank-reduced `extract_slice ... to tensor<4xf32>` + `expand_shape` | **무변화** — populateFoldUnitExtentDimsViaSlicesPatterns(:782-796) 에 이 패턴이 없음 |
| `@insert_unit` | RankReducedInsertSliceOp(:724-757): source `collapse_shape` + rank-reduced `insert_slice` | **무변화** — 동일 사유 |

→ slices 모드의 출력이 패턴 *목록* 차이(populate 분기)를 그대로 드러낸다.

### input/negative.mlir (negative: 발화 없음)

unit dim·상수0 접근·pad·slice 없음. dropUnitDims 는
`newIndexingMaps == indexingMaps` → failure (:489-493),
MoveInitOperandsToInput 은 body 가 `%out` 을 안 읽어 candidates 공집합 →
failure (:100-101). 출력 = pass 없이 통과시킨 IR 과 동일 (no-op 확인 완료).

## 파일

```
input/broadcast.mlir        # DropUnitDims: 상수0 접근 정리 + unit loop drop + linalg.index 시프트
input/unit-reduction.mlir   # DropUnitDims + MoveInitOperandsToInput 합주
input/pad-slice.mlir        # DropPadUnitDims + RankReduced{Extract,Insert}SliceOp (+모드 분기)
input/negative.mlir         # 발화 없음 (no-op)
output/output.<name>.mlir         # my-fold-unit-extent-dims (기본 모드)
output/intree.<name>.mlir         # linalg-fold-unit-extent-dims (기본 모드)
output/output.slices.<name>.mlir  # my-…=use-rank-reducing-slices=true
output/intree.slices.<name>.mlir  # linalg-…=use-rank-reducing-slices=true
run.sh                      # 전체 재현 + byte-diff 검증
```
