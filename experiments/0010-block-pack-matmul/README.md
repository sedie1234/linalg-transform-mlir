# 0010 block-pack-matmul — in-tree `linalg-block-pack-matmul` 해부·재현·관찰

in-tree pass `linalg-block-pack-matmul` (LLVM 19.1.7,
`mlir/lib/Dialect/Linalg/Transforms/BlockPackMatmul.cpp`) 를 해부하고,
out-of-tree pass `my-block-pack-matmul`
(`out-of-tree/lib/Passes/MyBlockPackMatmul.cpp`) 로 같은 절차를 재현해
byte-diff 로 검증한 실험.

## 재현

```bash
./run.sh   # 8 combo 전부 [OK byte-identical] 이면 성공 (exit 0)
```

## A. 호출 체인 해부 (파일:라인 = LLVM 19.1.7)

```
LinalgBlockPackMatmul : impl::LinalgBlockPackMatmulBase   BlockPackMatmul.cpp:279-307
  (def: Passes.td:139-196 — Pass<"linalg-block-pack-matmul">, anchor 없음.
   dependentDialects = [linalg, tensor] (Passes.td:172). 옵션 8개 :173-195)
  └─ runOnOperation()                                     BlockPackMatmul.cpp:283-306
       ├─ ControlBlockPackMatmulFn controlFn = lambda     :287-301
       │    옵션 8개 → BlockPackMatmulOptions (Transforms.h:1185-1211) 복사.
       │    mnkOrder 만 빈 리스트면 기본 {0,1,2} 유지 (:294-295).
       ├─ populateBlockPackMatmulPatterns(patterns, controlFn)        :303
       │    (선언 Transforms.h:1742-1743, 정의 BlockPackMatmul.cpp:310-320)
       │    └─ BlockPackMatmul<OpTy> ×7: GenericOp 전문화(:236-276) +
       │       Matmul/BatchMatmul/MatmulTransposeA/BatchMatmulTransposeA/
       │       MatmulTransposeB/BatchMatmulTransposeB (primary :217-234)
       │       matchAndRewrite(:223-230, generic 은 :244-272 — contraction
       │       interface + 3 가지 map 형태 {(i,k),(k,j)} {(k,i),(k,j)}
       │       {(i,k),(j,k)} → (i,j) 만 :261-263) → 위임:
       │       └─ linalg::blockPackMatmul                 BlockPackMatmul.cpp:138-214
       │            (선언 Transforms.h:1241-1243)
       │            1. hasPureBufferSemantics → fail      :141-142
       │            2. controlFn(op) → options            :144-146
       │            3. blockFactors.size() != 3 → fail    :148-149  ★ no-op 근거
       │            4. !allowPadding → validateFullTilesOnDims(:44-86) :154-159
       │            5. packMatmulGreedily                 Transforms.cpp:768-898
       │                 inferContractionDims (LinalgInterfaces.cpp:372)
       │                 → generalizeNamedOp(named 이면, :829-835)
       │                 → interchangeGenericOp((k,m,n) most-minor 정규화 :837-848)
       │                 → mnk-padded-multiples 시 affine ceilDiv 올림 (:874-888)
       │                 → linalg::pack (Transforms.cpp:480-610):
       │                   tensor.pack ×3 (+ 필요시 padding_value) +
       │                   packed linalg.generic + tensor.unpack ×1
       │            6. inferContractionDims(packed op)    :180-183
       │            7. transposePackedMatmul ×2 (LHS :190-199 / RHS :202-211,
       │               본체 :89-135) — 현재 layout 의 transposed 여부 판정
       │               (:107-110) 과 옵션이 다를 때만 perm={1,0} →
       │               packTranspose (Transforms.cpp:677-755):
       │               pack 의 outer_dims_perm/inner_dims_perm + generic 의
       │               indexing map 동시 재작성
       └─ applyPatternsAndFoldGreedily(op, patterns)      :304   ← greedy driver
```

핵심 함수 시그니처 (`mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h`):

| 함수 | 선언 | 정의 |
|---|---|---|
| `FailureOr<PackResult> blockPackMatmul(RewriterBase&, LinalgOp, const ControlBlockPackMatmulFn&)` | Transforms.h:1241-1243 | BlockPackMatmul.cpp:138-214 |
| `void populateBlockPackMatmulPatterns(RewritePatternSet&, const ControlBlockPackMatmulFn&)` | Transforms.h:1742-1743 | BlockPackMatmul.cpp:310-320 |
| `FailureOr<PackResult> packMatmulGreedily(RewriterBase&, LinalgOp, ArrayRef<OpFoldResult>, ArrayRef<int64_t>, ArrayRef<int64_t>)` | Transforms.h:1179-1183 | Transforms.cpp:768-898 |
| `FailureOr<PackResult> pack(RewriterBase&, LinalgOp, ArrayRef<OpFoldResult>)` | Transforms.h:1149-1150 | Transforms.cpp:480-610 |
| `FailureOr<PackTransposeResult> packTranspose(RewriterBase&, tensor::PackOp, LinalgOp, tensor::UnPackOp, ArrayRef<int64_t>, ArrayRef<int64_t>)` | Transforms.h:1166-1169 | Transforms.cpp:677-755 |
| `using ControlBlockPackMatmulFn = std::function<std::optional<BlockPackMatmulOptions>(LinalgOp)>` | Transforms.h:1217-1218 | — |

옵션 8개의 흐름: tablegen 멤버 → `runOnOperation` 의 controlFn lambda 가
`BlockPackMatmulOptions` 로 복사(:287-301) → pattern 이 op 마다
`blockPackMatmul` 안에서 `controlPackMatmul(linalgOp)` 호출(:144) 로 수신.
**block-factors 미지정 시 no-op**: 기본값이 빈 리스트 →
`options->blockFactors.size() != 3` (:148) → `"require 3 tile factors"`
matchFailure → greedy 가 IR 을 그대로 둔다 (이전 cycle 의 관찰과 일치).

## B. byte-diff 검증

`run.sh` 의 8 combo (입력 4 × 옵션 변형) 전부 `output.* == intree.*`
**byte-identical** — 이식 검증 완료.

## C. 코드 단계 ↔ IR 변화 매핑

### `bf.matmul` (block-factors=32,16,64, 기본 transpose 옵션)

64x256 · 256x128 matmul → blocked 4D. 6D iteration space
(d0=MB, d1=NB, d2=KB, d3=mb, d4=nb, d5=kb):

| IR 변화 | 만든 코드 단계 |
|---|---|
| `linalg.matmul` 소멸, `linalg.generic`(6D, par,par,red,par,par,red) 등장 | `packMatmulGreedily` 내 `generalizeNamedOp`(Transforms.cpp:829-835) + `pack`(:480-610) 이 named op 를 generic 으로 풀고 4 dim 을 3 minor dim (mb,nb,kb) 으로 분할 |
| `%pack = tensor.pack %A outer_dims_perm=[0,1] inner_dims_pos=[0,1] inner_tiles=[32,64]` → `2x4x32x64` = [MB][KB][mb][kb], map `(d0,d2,d3,d5)` | `pack`(Transforms.cpp:480-610) 의 LHS packing. 기본 `lhs-transpose-*=false` 라 `transposePackedMatmul` 의 LHS perm 은 identity (BlockPackMatmul.cpp:107-119 — 이미 [MB][KB] 인 layout 이 목표와 일치) |
| `%pack_0 = tensor.pack %B outer_dims_perm=[1,0] inner_dims_pos=[1,0] inner_tiles=[16,64]` → `8x4x16x64` = [NB][KB][nb][kb], map `(d1,d2,d4,d5)` | RHS 는 원래 [K][N] → 자연 packing 은 [KB][NB][kb][nb]. 기본 `rhs-transpose-outer/inner=true` 가 `transposePackedMatmul`(:202-211) → `packTranspose`(Transforms.cpp:677-755) 로 outer/inner perm 을 [1,0] 으로 재작성 — **mmt4d 형의 결정 지점** |
| `%pack_1 = tensor.pack %C inner_dims_pos=[0,1] inner_tiles=[32,16]` → `2x8x32x16` = [MB][NB][mb][nb], map `(d0,d1,d3,d4)` | output 은 transpose 옵션 대상이 아님 — `pack` 의 결과 그대로 (outer_dims_perm 생략 = identity) |
| `%unpack = tensor.unpack %3 ... into %arg2` → `64x128` 복원 | `pack` Step 4 (Transforms.cpp:587-601) — packed init 에 대칭인 UnPackOp 생성 |

결과 접근 패턴 = `[MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb]`
(linalg.mmt4d 와 동일).

### `bf-lhstrans.matmul` (+lhs-transpose-outer/inner=true)

LHS pack 만 `outer_dims_perm=[1,0] inner_dims_pos=[1,0] inner_tiles=[64,32]`
→ `4x2x64x32` = [KB][MB][kb][mb], map `(d2,d0,d5,d3)`. —
`transposePackedMatmul` LHS 경로(:190-199)에서 `isOuterTransposed(false) !=
transposeOuterBlocks(true)` → perm={1,0} (:114-119) → `packTranspose`.

### `bf-plain.matmul` (rhs-transpose-outer/inner=false)

RHS pack 이 `outer_dims_perm=[0,1] inner_dims_pos=[0,1] inner_tiles=[64,16]`
→ `4x8x64x16` = [KB][NB][kb][nb], map `(d2,d1,d5,d4)`. RHS transpose 옵션을
끄면 자연 packing 그대로 — 기본값과의 대조로 옵션→`packTranspose` 경로 확인.

### `bf8.generic-transpose-b` (generic, maps {(i,k),(j,k),(i,j)})

`BlockPackMatmul<GenericOp>` 전문화(:236-276)가 발화. B 가 이미 [N][K] 라
RHS 자연 packing 이 곧 [NB][KB][nb][kb] — `isOuterTransposed==true ==
rhsTransposeOuterBlocks(true)` → perm identity (:114-119), **재transpose 없이**
`outer_dims_perm=[0,1]` 로 끝난다. 결과 generic 의 map 은 bf.matmul 과 동일.

### `bf16.pad-matmul` (30x30x30, block-factors=16,16,16)

`%cst = arith.constant 0.0` + 모든 `tensor.pack` 에 `padding_value(%cst)` →
`2x2x16x16` (30 → 32 로 올려 packing). — `pack`(Transforms.cpp:553-572) 의
분기: `tensor::PackOp::requirePaddingValue(...)` 가 true 면 (:557-561 조건
불충족) else 가지(:564-571)에서 `getZeroAttr` 상수를 만들어 padded pack 생성.

### no-op 3종 (출력 = 입력 roundtrip, `tensor.pack` 0개)

| combo | no-op 근거 (코드) |
|---|---|
| `noopts.matmul` (옵션 없음) | blockFactors 빈 리스트 → `size() != 3` (BlockPackMatmul.cpp:148-149) |
| `nopad.pad-matmul` (allow-padding=false) | `validateFullTilesOnDims`(:44-86) — 30 % 16 != 0 → "expect packing full tiles only" (:154-159) |
| `bf16.negative-memref` (memref) | `hasPureBufferSemantics` → "require tensor semantics" (:141-142) |

## 파일

- `input/*.mlir` — 입력 4개 (positive 3 + negative 1, 각 파일 머리 주석 참조)
- `output/output.<combo>.<name>.mlir` — `--my-block-pack-matmul` 출력
- `output/intree.<combo>.<name>.mlir` — `--linalg-block-pack-matmul` 출력
- `run.sh` — 전체 재현 + byte-diff
