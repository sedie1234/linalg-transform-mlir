# #0009 linalg-to-loops-trio — in-tree `convert-linalg-to-{loops,affine-loops,parallel-loops}` 해부·재현·관찰

in-tree pass 3종 `convert-linalg-to-loops` / `convert-linalg-to-affine-loops` /
`convert-linalg-to-parallel-loops` (LLVM 19.1.7,
`mlir/lib/Dialect/Linalg/Transforms/Loops.cpp`) 를 해부하고, out-of-tree pass
`my-linalg-to-loops-trio` (`out-of-tree/lib/Passes/MyLinalgToLoopsTrio.cpp`,
옵션 `mode=scf|affine|parallel`) **1개** 로 같은 절차를 재현해 모드별
byte-diff 로 이식을 검증한 실험.

세 pass 는 **한 template 기계 `linalgOpToLoopsImpl<LoopTy>` 의 3
instantiation** 이다 — pass 별 차이는 LoopTy(`scf::ForOp` /
`affine::AffineForOp` / `scf::ParallelOp`) 와 dependentDialects 뿐. 변환
자체는 buffer-semantics linalg op 의 의미 3요소를 명시적 loop nest 로
푸는 것: `iterator_types` → loop 종류, `indexing_maps` → load/store index,
region → innermost body.

## 호출 체인 (해부 결과 — 파일:라인은 LLVM 19.1.7)

```
LowerToLoops          (scf.for)      Loops.cpp:339-348  ← --convert-linalg-to-loops
LowerToAffineLoops    (affine.for)   Loops.cpp:327-337  ← --convert-linalg-to-affine-loops
LowerToParallelLoops  (scf.parallel) Loops.cpp:350-357  ← --convert-linalg-to-parallel-loops
  def: Passes.td:33-47 / :26-31 / :49-59 — 셋 다 anchor 없는 op-agnostic
       Pass<"convert-linalg-to-*">, **옵션 0개**.
       dependentDialects(td) = {linalg,scf,affine} / {affine,linalg,memref}
       / {affine,linalg,memref,scf} + C++ override 가 memref(+scf) 추가
       (:331-333, :342-344)
└─ runOnOperation() → lowerLinalgToLoopsImpl<LoopType>(getOperation())
                                                       Loops.cpp:314-325
     ├─ patterns.add<LinalgRewritePattern<LoopType>>            :318
     │    (file-local, :257-275, MatchAnyOpTypeTag + benefit 1)
     │    matchAndRewrite                                       :263-274
     │      · dyn_cast<LinalgOp> 실패 ∨ !hasPureBufferSemantics()
     │        → notifyMatchFailure "expected linalg op with buffer
     │          semantics"                                      :266-269
     │      · linalgOpToLoopsImpl<LoopType>(rewriter, linalgOp) :270
     │      · 성공 시 rewriter.eraseOp(op)                      :272
     ├─ memref::DimOp / tensor::DimOp getCanonicalizationPatterns :319-320
     ├─ affine::AffineApplyOp::getCanonicalizationPatterns      :321
     ├─ patterns.add<FoldAffineOp>                              :322
     │    (file-local :287-312 — single-result & ≤1-input map 의
     │     affine.apply 를 상수(arith.constant index)/유일 operand 로 fold)
     └─ applyPatternsAndFoldGreedily(op, std::move(patterns))   :324
          → **greedy driver** (GreedyRewriteConfig 기본값)

linalgOpToLoopsImpl<LoopTy>                            Loops.cpp:208-254
  ① LoadOpTy/StoreOpTy 선택                            :211-216
     LoopTy==affine::AffineForOp → affine.load/affine.store
     그 외(scf.for/scf.parallel) → memref.load/memref.store
  ② loopRanges = linalgOp.createLoopRanges(rewriter, loc)  :223
     (LinalgInterfaces.cpp:994-1009 — getLoopsToShapesMap() 의
      AffineDimExpr 자리마다 Range{0, viewSizes[idx], 1};
      viewSizes 는 operand dim → dynamic 이면 memref.dim 생성)
  ③ GenerateLoopNest<LoopTy>::doit(…, bodyBuilderFn)   :227-236
     선언 Utils.h:356-365, 3 specialization:
       scf::ForOp        Utils.cpp:313-353 — scf::buildLoopNest
       AffineForOp       Utils.cpp:356-385 — affine::buildAffineLoopNest
                         (step 은 반드시 상수 — getConstantIntValue assert)
       scf::ParallelOp   Utils.cpp:523-567 → generateParallelLoopNest
                         (Utils.cpp:408-520, 재귀) — **연속 parallel
                         iterator 묶음 → 1개의 multi-iv scf.parallel,
                         reduction iterator → scf.for** 로 분리.
                         iterator_types 가 loop 종류를 가르는 유일 지점
     innermost body 에서 emitScalarImplementation<LoadOpTy,StoreOpTy>
                                                       Loops.cpp:127-175
       1.a input operand: scalar 면 그대로, 아니면
           makeCanonicalAffineApplies(b, loc,
             getMatchingIndexingMap(operand), allIvs)  :141-150
           (:39-56 — map 의 result expr 1개당 canonicalize 후
            affine.apply 1개 생성) → LoadOpTy
       1.b output operand 도 load (DPS init 읽기)       :152-158
       2-3. inlineRegionAndEmitStore                    :58-77
           region 을 IRMapping 으로 clone-inline,
           terminator(yield) operand → StoreOpTy
  ④ replaceIndexOpsByInductionVariables                :179-206 (호출 :252)
     loop nest 에서 iv 수집(scf.parallel 은 getInductionVars 복수)
     → 각 linalg.index dim 을 대응 iv 로 replaceOp

export 함수 (이식 시 가져다 쓸 수 있는 유일한 표면 — pattern/populate 없음):
  FailureOr<LinalgLoops> linalgOpToLoops(RewriterBase&, LinalgOp)
                                  선언 Transforms.h:769-770, 정의 Loops.cpp:368-371
  FailureOr<LinalgLoops> linalgOpToParallelLoops(…)
                                  선언 Transforms.h:773-774, 정의 Loops.cpp:374-378
  FailureOr<LinalgLoops> linalgOpToAffineLoops(…)
                                  선언 Transforms.h:777-778, 정의 Loops.cpp:362-365
  (셋 다 linalgOpToLoopsImpl<LoopTy> 호출 한 줄짜리 wrapper —
   in-tree pattern 의 :270 직접 호출과 동일 코드 경로)
```

### 특이점 (이식 시 주의)

1. **populate* 함수가 없는 pass**: `LinalgRewritePattern<LoopTy>` 와
   `FoldAffineOp` 는 둘 다 Loops.cpp file-local. 대신 알고리즘 본체가
   `linalgOpTo{Loops,AffineLoops,ParallelLoops}` 3개 함수로 export 된다
   (Transforms.h:769-778). 재현 = export 함수 호출 + 얇은 pattern 골격
   복제(가드/eraseOp 7줄) + FoldAffineOp verbatim 복제(25줄).
2. **load/store dialect 가 LoopTy 에 종속** (:211-216): affine 모드만
   affine.load/store — affine dialect 의 "index 는 affine map 결과" 제약을
   op 단위로 보존해, 이후 affine dependence 분석이 가능한 IR 을 만든다.
   scf 두 모드는 memref.load/store (아무 index 허용).
3. **mode=parallel 은 "전부 parallel" 이 아니다**: generateParallelLoopNest
   (Utils.cpp:408-520) 가 iterator_types 를 앞에서부터 훑으며 연속 parallel
   구간만 multi-iv scf.parallel 하나로 묶고, reduction 을 만나면 scf.for 로
   떨어진다 (matmul: (i,j) parallel + k for). **합법성 판단이 아니라
   linalg op 에 이미 선언된 iterator_types 를 그대로 믿는다** — affine 의
   isLoopParallel(의존성 분석) 과 대조되는 지점.
4. **affine 모드의 step 상수 assert** (Utils.cpp:370-377):
   buildAffineLoopNest 은 step 이 컴파일타임 상수여야 한다. linalg op 의
   loop range 는 createLoopRanges 가 항상 step=1 로 만들므로 (LinalgInterfaces.cpp:1005)
   이 pass 경유로는 안전.
5. **identity 접근의 affine.apply 가 IR 에 안 남는 이유**: ①
   makeCanonicalAffineApplies 의 canonicalizeMapAndOperands + ② 같은
   greedy 에 등록된 FoldAffineOp(:287-312)/AffineApplyOp canonicalization
   (:321) 이 d0→d0 류 apply 를 즉시 fold → iv 가 직접 load/store index 로.
   비-trivial map (예: conv 의 d0+d1) 일 때만 affine.apply 가 남는다.
6. **tensor semantics 는 전제조건 위반으로 그냥 비발화** (에러 아님):
   가드 :266-269 가 notifyMatchFailure 만 반환 — pass 는 성공으로 끝나고
   IR 불변. bufferization 이 반드시 선행해야 한다 (Passes.td:36-41).

## 코드 단계 ↔ IR 변화 매핑

### 1) `input/matmul.mlir` — named op, dynamic shape (3 모드 발화)

before (공통):

```mlir
linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
              outs(%C : memref<?x?xf32>)
```

| IR 변화 (after 에 나타난 것) | 만든 코드 |
|---|---|
| `%dim = memref.dim %arg0, %c0` 등 3개 (loop ub 재료) | `createLoopRanges` LinalgInterfaces.cpp:994-1009 — matmul 의 loopsToShapesMap `(d0,d1,d2)->(d0,d2,d2,d1,d0,d1)` 의 첫 등장 dim 마다 operand dim 추출 (이때 d0←A#0, d2←A#1, d1←B#1 이라 ub 순서가 %dim, %dim_1, %dim_0) |
| [scf] 3중 `scf.for %arg3/%arg4/%arg5` | `GenerateLoopNest<scf::ForOp>::doit` Utils.cpp:313-353 (scf::buildLoopNest) |
| [affine] 3중 `affine.for … = 0 to %dim` | `GenerateLoopNest<AffineForOp>::doit` Utils.cpp:356-385 — bound 가 attr 아닌 affine bound (`0 to %dim`), `%c0/%c1` 은 bound 로 안 쓰여도 memref.dim 의 index 로 잔존 |
| [parallel] `scf.parallel (%arg3,%arg4) = (%c0,%c0) to (%dim,%dim_1)` + 안에 `scf.for %arg5` + `scf.reduce` | `generateParallelLoopNest` Utils.cpp:408-520 — iterator_types=[par,par,red] 의 앞 연속 par 2개를 multi-iv scf.parallel 1개로, red 는 scf.for 로 |
| `memref.load %arg0[%arg3, %arg5]` / `…%arg1[%arg5, %arg4]` / `…%arg2[%arg3, %arg4]` ([affine] 은 affine.load) | `emitScalarImplementation` Loops.cpp:141-158 — indexing_maps (d0,d2)/(d2,d1)/(d0,d1) 을 `makeCanonicalAffineApplies` :39-56 로 전개. identity 접근이라 affine.apply 는 FoldAffineOp/:321 canonicalization 에 전부 fold → iv 직접 사용 |
| `arith.mulf` + `arith.addf` (matmul 의 암묵 region) | `inlineRegionAndEmitStore` Loops.cpp:58-77 — region clone-inline |
| `memref.store %4, %arg2[%arg3, %arg4]` ([affine] 은 affine.store) | 같은 함수 :71-76 — yield operand 를 outs buffer 에 store |
| `linalg.matmul` 소멸 | `LinalgRewritePattern::matchAndRewrite` :272 의 eraseOp |

### 2) `input/generic-index.mlir` — generic + linalg.index, par+red 혼합

before 핵심: `iterator_types = ["parallel","reduction"]`, body 에
`linalg.index 0/1` 사용 (out[i] += in[i][j] + (i+j)).

| IR 변화 | 만든 코드 |
|---|---|
| [parallel] `scf.parallel (%arg2) = (%c0) to (%c4)` **안에** `scf.for %arg3 = %c0 to %c8` — d0 만 parallel, d1 은 for | `generateParallelLoopNest` Utils.cpp:428-440 — front 가 parallel 아니면 buildLoopNest 로 sequential 1개 생성 후 재귀 (:460-476 DistributionMethod::None 의 parallel 묶음 생성과 분리) |
| [affine] `affine.for %arg2 = 0 to 4` — static shape 이라 bound 가 순수 상수, `%c0/%c4` 류 arith.constant 가 IR 에 하나도 안 남음 | `createLoopRanges` 가 만든 `Range{0,4,1}` 의 attr bound 를 buildAffineLoopNest 이 affine bound 로 직접 수용 |
| `%2 = arith.addi %arg2, %arg3` — body 의 `linalg.index 0/1` 이 실제 loop iv 로 치환 | `replaceIndexOpsByInductionVariables` Loops.cpp:179-206 — scf.parallel 은 `getInductionVars()` 복수 수집 (:186-188) 후 `allIvs[indexOp.getDim()]` 로 replaceOp (:200-205) |
| `%1 = memref.load %arg1[%arg2]` (output 도 load) → 누적 후 같은 자리에 store | `emitScalarImplementation` 1.b :152-158 — **reduction 은 DPS init 읽기-수정-쓰기로 표현**, scf.reduce 연산으로 만들지 않는다 (scf.reduce 는 빈 terminator) |

### 3) `input/tensor-negative.mlir` — negative (3 모드 비발화)

`linalg.matmul ... -> tensor<4x4xf32>` (tensor semantics).
`hasPureBufferSemantics()` false → 가드 Loops.cpp:266-269 의
notifyMatchFailure → pattern 비발화 → **세 모드 모두 IR 불변**
(byte-diff 로 in-tree 와 동일 불변임을 확인). #0008 의 negative 와 달리
이 pass 의 greedy 에는 region simplification 으로 모양이 변할 CFG 도
없어 입력 == 출력.

## byte-diff 검증 결과

`run.sh` 가 자동 검증. 결과: **13/13 byte-identical**

- 본 실험 입력 3종 × 3모드 = 9 케이스: `output.<mode>.<n>.mlir` ==
  `intree.<mode>.<n>.mlir`
- in-tree 회귀 테스트 교차 검증 4 케이스: `loops.mlir`(scf, parallel) /
  `affine.mlir`(affine) / `parallel-loops.mlir`(parallel) 전부 일치

## 파일

```
input/matmul.mlir           named op + dynamic shape (3 모드 발화)
input/generic-index.mlir    generic, par+red 혼합 + linalg.index (3 모드 발화)
input/tensor-negative.mlir  tensor semantics matmul (3 모드 비발화 negative)
output/output.<mode>.<n>.mlir  my-linalg-to-loops-trio{mode=<mode>} 출력
output/intree.<mode>.<n>.mlir  convert-linalg-to-{loops,affine-loops,parallel-loops} 출력
run.sh                      전체 재현 + byte-diff (in-tree 테스트 4케이스 포함)
```

재현: `./run.sh` (전제: `out-of-tree/build/bin/my-mlir-opt` 빌드 완료)
