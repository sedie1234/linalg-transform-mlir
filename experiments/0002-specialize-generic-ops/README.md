# 0002 — linalg-specialize-generic-ops 해부·재현·관찰

in-tree pass `linalg-specialize-generic-ops` (`linalg.generic` → named op)를
해부(A)하고, out-of-tree pass `my-specialize-generic-ops`로 재현(B)하고,
IR 전후 변화를 관찰(C)한 실험. #0001 generalize 의 정확한 역방향.
상세 로그: `../../log/0002-specialize-generic-ops.html`.

## 재현

```bash
./run.sh
# 각 input/*.mlir 에 대해 output/output.<name>.mlir (my pass) 와
# output/intree.<name>.mlir (in-tree pass) 를 캡처하고 byte-diff.
# 결과: 3/3 byte-identical (2026-06-12, LLVM 19.1.7)
```

## A. 호출 체인 (파일:라인 — LLVM 19.1.7)

```
Passes.td:97-100        def LinalgSpecializeGenericOpsPass : Pass<"linalg-specialize-generic-ops">
                        (옵션 0개, dependentDialects = ["linalg::LinalgDialect"], anchor 없음)
Specialize.cpp:312-319  struct LinalgSpecializeGenericOpsPass : impl::...PassBase<...>
Specialize.cpp:322-328  runOnOperation()
  ├─ :324 populateLinalgGenericOpsSpecializationPatterns(patterns)
  │         정의 Specialize.cpp:330-333 / 선언 Transforms.h:1596
  │         └─ patterns.add<LinalgSpecializationPattern>(ctx)
  │              Transforms.h:1425-1437  (OpRewritePattern<GenericOp>)
  │              └─ matchAndRewrite → returningMatchAndRewrite
  │                   └─ specializeGenericOp(rewriter, op)
  │                        선언 Transforms.h:698 / 정의 Specialize.cpp:262-309
  │                        ── idiom 인식 분기 (이 순서대로) ──
  │                        ├─ isaCopyOpInterface        :264 → linalg.copy
  │                        │    (LinalgInterfaces.cpp:56-71 — 전 loop parallel
  │                        │     + 1-in/1-out + maps 모두 identity + body 가 yield 단독)
  │                        ├─ isaFillOpInterface        :270 → linalg.fill
  │                        │    (LinalgInterfaces.cpp:76-100 — scalar 입력 +
  │                        │     body 가 yield(arg0) 단독)
  │                        ├─ isaElemwiseSingleUnaryOpInterface :276
  │                        │    (LinalgInterfaces.cpp:142-151 — body 2개 op
  │                        │     [unary+yield], maps identity) — math.exp 만
  │                        │    → REPLACE_UNARY_OP(ExpOp)  매크로 :39-42
  │                        ├─ isaElemwiseSingleBinaryOpInterface :284
  │                        │    (LinalgInterfaces.cpp:153-164)
  │                        │    ├─ areBinOpsSwapped :58-69 — body 첫 op 의
  │                        │    │   operand 0 이 block arg 0 인지로 swap 판정
  │                        │    └─ arith.{addf,subf,mulf,divf} → linalg.{add,sub,mul,div}
  │                        │        REPLACE_BINARY_OP 매크로 :32-37 (swap 시 ins 교환)
  │                        └─ isaContractionOpInterface :305
  │                             (LinalgInterfaces.cpp:453-460)
  │                             └─ specializeLinalgContractions :148-255
  │                                  ├─ 2-in/1-out 검사            :150
  │                                  ├─ projectedPermutation 검사  :154-157
  │                                  ├─ inferContractionDims       :178
  │                                  │    (LinalgInterfaces.cpp:371-377 → Impl :328-369)
  │                                  ├─ m/n/k 각각 정확히 1개      :182-183
  │                                  ├─ isContractionBody          :185-193
  │                                  │    (LinalgInterfaces.cpp:186-248;
  │                                  │     mulf+addf / muli+addi / complex 페어만)
  │                                  ├─ map rank = batch+2         :196-205
  │                                  ├─ batch dim identity 검사    :207-221
  │                                  ├─ matchOperandMap(A/B/C)     :223-228
  │                                  │    (:111-133 — Match/Transposed/Mismatch)
  │                                  ├─ C==Match 필수, A·B 동시 Transposed 금지 :235-237
  │                                  └─ replaceWithMatmulVariant<T> :139-145
  │                                       → linalg.{batch_}matmul{_transpose_a|_b} :240-254
  └─ :326 applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
          ← driver = greedy (미수렴 시 signalPassFailure, in-tree 동일)
```

## B. out-of-tree 재현 (4-edit)

| 파일 | 변경 |
|------|------|
| `out-of-tree/lib/Passes/MySpecializeGenericOps.cpp` | 신규 — in-tree runOnOperation 과 동일 절차 (populate + greedy + signalPassFailure) |
| `out-of-tree/lib/Passes/CMakeLists.txt` | SOURCES 에 한 줄 추가 |
| `out-of-tree/lib/Passes/PassRegistration.cpp` | `registerPass(createMySpecializeGenericOpsPass)` |
| `out-of-tree/include/MyPasses/Passes.h` | factory 선언 + 호출 체인 요약 주석 |

link libs 는 기존 `MLIRLinalgTransforms`(populate/specializeGenericOp 정의) +
`MLIRLinalgDialect`(isa*OpInterface 판정 함수들, LinalgInterfaces.cpp) +
`MLIRTransforms`(greedy driver)로 충분 — 추가 없음.

## C. 코드 단계 ↔ IR 변화 매핑

| 입력 | 함수 | 코드 단계 (Specialize.cpp) | IR 변화 |
|------|------|---------------------------|---------|
| `elemwise-copy-fill.mlir` | `@copy_2d` | `isaCopyOpInterface` :264 — maps identity×2 + yield 단독 body | generic(7줄) → `linalg.copy ins outs` 1줄. maps/iterators/body 속성 전부 암묵화 |
| 〃 | `@fill_2d` | `isaFillOpInterface` :270 — scalar 입력 + yield(arg0) | generic → `linalg.fill ins(%cst : f32)`. scalar map `()` 소멸 |
| 〃 | `@exp_2d` | `isaElemwiseSingleUnaryOpInterface` :276 + `isa<math::ExpOp>` :278 | generic(body `math.exp`) → `linalg.exp` |
| 〃 | `@add_2d` | `isaElemwiseSingleBinaryOpInterface` :284 + `isa<arith::AddFOp>` :287 | generic(body `arith.addf %in, %in_0`) → `linalg.add` |
| 〃 | `@sub_swapped` | `areBinOpsSwapped` :58-69 → swap=true → REPLACE_BINARY_OP :32-37 | body 가 `subf %in_0, %in` (역순) → **`linalg.sub ins(%arg1, %arg0)`** — named op 의 operand 순서가 교환되어 의미 보존 |
| 〃 | `@div_2d` | `isa<arith::DivFOp>` :299 | generic → `linalg.div` |
| `contraction-variants.mlir` | `@matmul_f32` | `matchOperandMap` A/B/C 모두 Match :223-228 → :254 | generic(mulf+addf body) → `linalg.matmul` |
| 〃 | `@matmul_transpose_b` | B map `(d1,d2)` → Transposed :252 | → `linalg.matmul_transpose_b` |
| 〃 | `@matmul_transpose_a` | A map `(d2,d0)` → Transposed :250 | → `linalg.matmul_transpose_a` |
| 〃 | `@batch_matmul_i32` | `dims.batch.size()==1` + batch identity 검사 :207-221 → :247 | muli+addi 정수 페어도 `isContractionBody` 페어 목록에 있어 발화 → `linalg.batch_matmul` |
| `negative-no-specialize.mlir` | `@transpose_like` | maps 비-identity → isaCopy/isaElemwise 모두 false. **transpose 인식 분기 자체가 없음** :262-308 | generic 잔류 (재인쇄만) |
| 〃 | `@fused_exp_neg` | body 2-op → `isaElemwiseSingle*` 의 size==2 검사 false (LinalgInterfaces.cpp:127-129) | generic 잔류 |
| 〃 | `@max_elemwise` | binary 인터페이스 통과하나 `arith.maximumf` 는 :287-302 화이트리스트 밖 | generic 잔류 |
| 〃 | `@multi_k_contract` | `dims.k.size()!=1` :182 reject | generic 잔류 |

핵심 통찰: **specialize 는 generalize 의 역이지만 전사(onto)가 아니다.**
generalize 는 모든 named op 에 무조건 발화하는 "정보 노출"인 반면, specialize 는
generic 의 구조를 *판정*해 인식 가능한 idiom 부분집합(copy/fill/exp/add/sub/mul/div/
matmul 6변형)에만 발화한다. round-trip 검증: 0001 입력(matmul+add+transpose)에
`--my-generalize-named-ops --my-specialize-generic-ops` 를 걸면 matmul/add 는 복원되고
transpose 는 generic 으로 잔류한다 (인식 분기 부재).

## byte-diff 결과

```
[OK ] byte-identical : contraction-variants
[OK ] byte-identical : elemwise-copy-fill
[OK ] byte-identical : negative-no-specialize
```
