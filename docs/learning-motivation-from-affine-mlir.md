# Linalg + Transform MLIR 학습 — affine-mlir에서 얻은 동기와 비교 필요성

> **2026-05-19 NAS KB 이관 완료.** 선행 워크스페이스 affine-mlir의 핵심 지식(#0001~#0007 + #0023 + 본 motivation 흐름)이 영구 지식 베이스 `~/NAS/__MyNeuron/neuron/`로 정제 이관되었다. 본 문서 정제판은 `대화/compiler/ai-compiler/mlir/linalg-transform-motivation-from-affine.md`, 워크스페이스 entry는 `프로젝트/affine-mlir/affine-mlir-overview.md`.

본 문서는 `/home/hwan/workspace/affine-mlir/`에서 누적된 polyhedral·affine 학습의 결과로 본 워크스페이스가 *왜 필요한가*, 그리고 affine 영역과 *무엇을 1:1 대조 학습해야 하는가*를 정리한다. 본 워크스페이스의 *첫 cycle 진입 전*에 한 번 정독.

대상 워크스페이스: `linalg-transform-mlir` (Plane: `linalgtransformmlir`, identifier LXT).
선행 워크스페이스: `affine-mlir` (Plane: `affinemlir`, identifier AFMLIR).

---

## 1. affine-mlir에서 누적된 학습 결과 (= 본 워크스페이스의 baseline)

### 1.1 학습 spine 한 줄

> **본인이 out-of-tree pass를 직접 작성·등록·실행한다.** in-tree API는 호출만, 알고리즘은 in-tree에 위임. legality 판정·정책 결정만 본인 작성.

(`/home/hwan/workspace/affine-mlir/LEARNING_PREMISE.md` §2.)

### 1.2 핵심 산출물 (work 단위)

| Work | Module | 내용 | 학습 정점 |
|---|---|---|---|
| **#0005** | M1-1 | `--my-dependence-check` — affine 분석 4 API 호출 | stencil 2D의 i축·j축 RAW가 *다른 depth에 분리* 보고됨 (wavefront 입력 데이터) |
| **#0006** | M2-3 | `--my-permute-if-legal` — 분석+변환 결합 spine 정점 | illegal swap 강제 시 RAW→WAR 라벨 뒤집힘 (의미 깨짐의 코드 증거). Allen-Kennedy criterion 구현 |
| **#0007** | M2-1 | `--my-unroll-by-N` — 첫 IR 변환 pass | 본인 pass·in-tree CLI·pipeline string **3중 byte-identical** |
| **#0008~#0022** | T1 (15 work) | 모든 in-tree affine pass의 효과 관찰 (pipeline string variant) | `affine-loop-tile` separate=true의 `affine.if` peeling / `affine-parallelize`의 reduce syntax / `lower-affine`의 mod Euclidean correction |
| **#0023** | ad-hoc | linalg.generic payload 실험 (본 워크스페이스의 직접 동기) | payload의 SSA tmp = intermediate buffer 0 byte (polyhedral array contraction의 MLIR 자동 형태) |

### 1.3 내재화된 *폴리헤드럴 ↔ affine* 매핑

| Polyhedral baseline | MLIR affine 표현 | 검증된 work |
|---|---|---|
| iteration domain D | `affine.for` 중첩 구조 (암묵), `FlatAffineValueConstraints` (명시) | #0002, (예정) M1-2 |
| access function A | `affine.load/store`의 `affine_map` indices | #0002, #0005 |
| dependence relation | `checkMemrefAccessDependence` Presburger feasibility | #0005, #0006 |
| schedule = permutation 행렬 P | `permuteLoops` + `permMap` | #0006 |
| schedule = unroll (1D strip-mining) | `loopUnrollByFactor` | #0007 |
| schedule = tile | `tilePerfectlyNested` (in-tree), `--affine-loop-tile` | #0013 |
| lex positivity (Allen-Kennedy) | `checkSwapLegalityOne` (본인 작성) / `isValidLoopInterchangePermutation` (in-tree) | #0006 |
| fusion = same schedule | `--affine-loop-fusion` (별개 loop → 한 loop) | #0015 |
| data-copy-generate (Pluto inspired) | `--affine-data-copy-generate` → memref + memory space attribute | #0016 |
| software pipelining (modulo) | `--affine-pipeline-data-transfer` | #0018 |

→ **polyhedral 절반은 코드 레벨로 박힘**. lex+ 검사·dep distance vector·schedule 행렬·tile peeling 모두 *실측 산출물로 누적*.

---

## 2. 왜 linalg + transform이 *그 다음* 단계인가

### 2.1 affine만으로 못 풀리는 문제 — *high-level semantic의 손실*

affine은 *명시적 좌표*가 있어야 분석. 그 결과:

| 문제 | affine의 한계 |
|---|---|
| conv를 *한 statement로* 유지 | 불가 — 7중 nest로 *분해된 후*에야 polyhedral 분석 |
| matmul → BLAS gemm (외부 호출) | 불가 — 분해된 후 *패턴 인식 손실* |
| conv → Winograd 변환 | 불가 — *named identity*가 사라짐 |
| Tensor Core / VNNI / SME 매핑 | 불가 — high-level instruction 인식 X |
| 데이터 layout 재배치 (AoS↔SoA, pack/unpack) | 어색 — affine은 *좌표 변환*만, layout은 데이터 차원 변환 |

본 워크스페이스 #0023 ad-hoc 실험이 *문제의 일면*을 보여줌:
- linalg.generic payload에 두 op을 묶으면 *intermediate buffer 0 byte* (polyhedral array contraction의 자동 형태)
- 같은 효과를 affine으로 표현하려면 *명시적 scalrep + dataflow 분석* 필요
- 즉 **linalg가 *high-level에서* 그 변환을 *자연스럽게* 제공**

### 2.2 *대화에서 사용자가 명시한 요구*

[/home/hwan/workspace/affine-mlir/ 의 대화 기록에서 추출]

1. **"conv나 matmul 같은 grouped data에도 affine 적용되나? conv를 하나의 statement로 두고 싶다"** → 답: affine 직접 불가. linalg.conv_2d named op이 정답.
2. **"고수준의 정보를 오래 가져가고 싶다"** → 답: linalg level에서 macro 변환, affine level에서 micro 변환 (progressive lowering).
3. **"linalg에서 데이터 fetch 순서 최적화 기법이 있나? polyhedral은 register-cache-sram-dram 계층 매핑이 가능한데"** → 답: linalg의 tile + promote + fuse + vectorize + distribute가 같은 영역. Transform dialect로 *명시적 schedule IR* 작성.
4. **"linalg와 transform dialect 공부가 필요"** → 본 워크스페이스 신설 결정.

→ **본 워크스페이스의 정확한 학습 의도**:
- conv/matmul 같은 *high-level op을 한 단위로* 유지하면서 *변환 (tile/fuse/promote/vectorize)*
- *명시적 schedule IR* (transform dialect)로 schedule 결정 (Halide-style)
- 그 결과 IR을 affine으로 lower해 *polyhedral micro 변환*과 *조합*
- 두 영역(polyhedral 자동 vs linalg+transform 명시)의 *결과 동등성·차이*를 코드 레벨 검증

---

## 3. affine ↔ linalg+transform 핵심 차이 (학습 비교 기준)

### 3.1 철학 차이

| 측면 | Polyhedral (affine) | Linalg + Transform |
|---|---|---|
| **모델 단위** | iteration domain + access function | indexing maps + iterator types (parallel/reduction) |
| **분석 도구** | Presburger / ILP (완전 자동) | interface 기반 + 보조 polyhedral (linalg 안의 IR-level 활용) |
| **schedule 결정** | Pluto-style 통합 ILP 풀이 | transform dialect로 *IR에 schedule 명시* (Halide style) |
| **High-level pattern** | ✗ (분해 후에만) | ✓ (matmul-as-gemm, conv-as-Winograd, contract-as-TensorCore) |
| **Data layout 변환** | 어색 | `linalg.pack/unpack`으로 1급 |
| **Auto-scheduler** | Pluto, isl-scheduler | IREE Auto-Scheduler, AKG, Triton |

### 3.2 같은 변환의 다른 표현 (1:1 대조 학습 과녁)

| 변환 효과 | affine 표현 (이미 학습) | linalg+transform 표현 (학습 대상) |
|---|---|---|
| **Tile (cache 매핑)** | `--affine-loop-tile` (#0013) | `transform.structured.tile_using_for` |
| **Unroll** | `--affine-loop-unroll` (#0011) | (linalg 직접 X — affine으로 lower 후) |
| **Interchange/permute** | `permuteLoops` (#0006) | `transform.structured.interchange` (iterator_types 재배치) |
| **Fusion (별개 loop 합치기)** | `--affine-loop-fusion` (#0015) | `transform.structured.fuse_into_containing_op` |
| **Fast memory copy** | `--affine-data-copy-generate` (#0016) | `transform.structured.promote` (operands_to_promote + memory_space) |
| **Vectorize** | `--affine-super-vectorize` (#0020) | `transform.structured.vectorize` (+ vector.contract) |
| **Parallelize** | `--affine-parallelize` (#0019) | `transform.structured.tile_using_forall` + GPU/thread mapping |
| **Layout 변환** | (어색) | `linalg.pack/unpack` |
| **Named op semantic** | (없음) | `linalg.matmul`, `linalg.conv_2d_*`, `linalg.generic` |

→ **각 행이 본 워크스페이스의 1 work 후보**. affine 산출물과 *결과 IR 비교 가능*.

### 3.3 progressive lowering의 정석

```
[High level — semantic preserved]
linalg.matmul (named op)
       ↓ transform dialect로 schedule 결정 (macro)
linalg.generic (tiled, promoted, vectorized form)
       ↓ --convert-linalg-to-affine-loops
[Mid level — polyhedral 가능]
affine.for + memref<…, #space>
       ↓ affine 변환 (micro — LICM, scalrep)
affine optimized
       ↓ --lower-affine
[Low level]
scf + memref + (vector)
       ↓ 본인 backend lowering
LLVM / EmitC / SPIR-V
```

본 워크스페이스에서 **linalg level (macro)** + **affine level (micro)** 양쪽을 학습. affine-mlir이 mid/low 절반을 깔아둔 셈.

---

## 4. 학습 트랙 제안 (orchestrator가 첫 cycle에서 합의)

### T1: Linalg named ops + generic

- T1-1: `linalg.matmul` IR 직접 작성 + `--linalg-generalize-named-ops`로 generic 변환
- T1-2: `linalg.conv_2d_nhwc_hwcf` named op + indexing maps 해석
- T1-3: `linalg.generic` payload 4단계 (single-op / multi-op / SSA-pipeline / reduction)
- T1-4: linalg → affine → scf full lowering chain 단계별 IR 캡처

### T2: Transform dialect — schedule decisions

- T2-1: `transform.structured.tile_using_for` — affine-mlir #0013 tile과 1:1 비교
- T2-2: `transform.structured.promote` + memory space — affine-mlir #0016과 비교
- T2-3: `transform.structured.fuse_into_containing_op` — affine-mlir #0015와 비교
- T2-4: `transform.structured.vectorize` + `vector.contract` matmul SIMD
- T2-5: `transform.structured.tile_using_forall` + GPU mapping
- T2-6: Schedule chain (tile → promote → vectorize → distribute) 명시적 작성

### T3: Linalg pattern lib

- T3-1: `linalg.pack/unpack` layout 변환 — AoS↔SoA 효과
- T3-2: matmul → vector.contract (Tensor Core/VNNI-style)
- T3-3: conv → im2col, Winograd 변환 (linalg 또는 본인 pass)

### T4: 본 워크스페이스의 spine — 본인 linalg/transform pass 작성

- T4-1: out-of-tree linalg utility 호출 pass (예: 본인 tile schedule)
- T4-2: 본인 transform op 작성 (외부 schedule script 소비)

→ T1 → T2가 핵심 학습. T3/T4는 심화.

---

## 5. affine-mlir과의 *직접 비교 학습 후보* (가장 학습 가치 큰 work들)

### (a) Tile 결과의 byte-equivalence

- affine-mlir `experiments/0013-t1-affine-loop-tile/output.*.mlir` (separate=true 포함)
- linalg-transform-mlir에서 `linalg.matmul` + `transform.tile_using_for` + lower
- *동일 입력 행렬에 대해 결과 affine IR이 일치하는가* 확인. linalg가 *fast path*로 가는 경우 vs *generic*으로 lower하는 경우 비교.

### (b) Fusion의 *동등성* — #0023 follow-up

- affine-mlir #0023 Case 02b가 *tensor*에서만 fuse-elementwise 성공. memref form은 no-op.
- transform dialect로 동일 fusion을 *명시적*으로 표현. + bufferize까지 가서 affine IR과 비교.
- *linalg fusion = polyhedral schedule clustering*의 완전한 코드 레벨 증거.

### (c) Promote vs data-copy-generate

- affine-mlir #0016 `--affine-data-copy-generate` 결과의 자동 copy nest + tag-memref
- linalg-transform-mlir의 `transform.promote` 결과 + bufferize + lower
- *같은 결과인가? 어디서 갈라지는가?* — Pluto inspired 자동 vs 명시적 schedule

### (d) Parallelize의 reduce syntax

- affine-mlir #0019 `affine.parallel ... reduce ("addf") -> (f32)` syntax
- linalg-transform-mlir의 `linalg.matmul` (자체 reduction iterator) → tile_using_forall + GPU mapping → lower
- reduction이 *어느 레벨에 박히는가* 비교

### (e) Vectorize 결과 비교

- affine-mlir #0020 `--affine-super-vectorize` 결과 IR
- linalg-transform-mlir의 `linalg.vectorize` + `vector.contract` lower
- 두 영역의 `vector` dialect 사용 패턴 비교

---

## 6. 운영 규칙

- **본 워크스페이스의 spine은 LEARNING_PREMISE에 따로 박을 것** (affine-mlir과 동일 — 본인 작성 위주). 단 linalg/transform 영역은 *transform dialect schedule script 자체가 본인 작성*의 핵심 형식. C++ pass보다 *transform IR* 작성이 dominant.
- **affine-mlir과의 비교 결과는 각 work의 README §4에 baseline 표로 박을 것**. 동일 입력·동일 의미 변환에 대해 두 영역 IR diff 또는 동등성 명시.
- **본 워크스페이스의 mlir-opt는 affine-mlir의 빌드 산출물 (`~/llvm-project/build/bin/mlir-opt`) 재사용 가능**. 별도 빌드 불필요. `experiments/0001-build-mlir-opt/env.sh` 참조하거나 직접 PATH 추가.
- **첫 cycle에서 orchestrator와 *T1·T2 트랙 module 합의* 후 Plane에 등재**.

---

## 7. 즉시 시작 절차

```bash
cd /home/hwan/workspace/learning-workspace/linalg-transform-mlir
source /home/hwan/workspace/affine-mlir/experiments/0001-build-mlir-opt/env.sh   # mlir-opt PATH 설정
# Claude Code에서:
# > "linalg-transform-mlir-orchestrator로 학습 시작할게. 본 docs/learning-motivation-from-affine-mlir.md 읽고 T1·T2 트랙 module 안 확정해주세요."
```

orchestrator가:
1. 본 문서 + CLAUDE.md 읽기
2. T1·T2 트랙의 module 후보 사용자와 합의
3. `learning-module-add` skill로 Plane (project `linalgtransformmlir`)에 module 등재
4. 첫 work (T1-1 `linalg.matmul` named op 작성) experimenter 디스패치

---

## 8. 핵심 한 줄

> **본 워크스페이스는 affine-mlir의 *직접 후속*.** polyhedral 절반은 affine-mlir에서 깔렸고, 이제 *high-level semantic 보존* 영역(linalg) + *명시적 schedule IR* 영역(transform)으로 확장. *같은 변환 효과*를 두 영역이 어떻게 다르게 표현·결정하는지 1:1 대조 학습이 spine. 비교 대상 산출물은 affine-mlir의 #0006·#0007·#0013·#0015·#0016·#0019·#0020·#0023.

## 9. 참고 경로

| 항목 | 경로 |
|---|---|
| 본 워크스페이스 | `/home/hwan/workspace/learning-workspace/linalg-transform-mlir/` |
| 선행 워크스페이스 (baseline) | `/home/hwan/workspace/affine-mlir/` |
| affine-mlir API 레퍼런스 | `/home/hwan/workspace/affine-mlir/docs/affine_api_reference.md` |
| affine-mlir 핵심 가이드 | `/home/hwan/workspace/affine-mlir/code-viewer/` (Dep check·Permute·Unroll 가이드 3종) |
| mlir-opt 빌드 산출물 | `~/llvm-project/build/bin/` |
| LLVM linalg 소스 | `~/llvm-project/mlir/lib/Dialect/Linalg/` |
| LLVM transform 소스 | `~/llvm-project/mlir/lib/Dialect/Transform/` |
| Plane 진입점 | `/home/hwan/workspace/plane-learning-space/.agent/` (project `linalgtransformmlir`, ID `201fab3a-35ac-4f28-8e66-a45f2b3c63ee`, identifier `LXT`) |
