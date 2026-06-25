#!/usr/bin/env bash
# #0010 block-pack-matmul — 재현 스크립트
#
# 각 (입력 × 옵션 조합)에 대해
#   output.<combo>.<name>.mlir : my-mlir-opt --my-block-pack-matmul=<opts>
#   intree.<combo>.<name>.mlir : my-mlir-opt --linalg-block-pack-matmul=<opts>
# 를 캡처하고 byte-diff 로 이식 정확성을 검증한다.
#
# in-tree pass 는 anchor 없는 op-agnostic pass 라 module 에 직접 올릴 수 있다.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="$DIR/../../out-of-tree/build/bin/my-mlir-opt"

mkdir -p "$DIR/output"

# (입력이름, combo 라벨, pass 옵션 문자열) 튜플.
# 옵션 문자열이 비면 옵션 없이 호출 (= block-factors 미지정 no-op 검증).
RUNS=(
  # positive 1 — named matmul, 깨끗하게 나눠떨어지는 packing.
  "matmul|noopts|"
  "matmul|bf|block-factors=32,16,64"
  # LHS outer+inner transpose 옵션 분기 (transposePackedMatmul LHS 경로).
  "matmul|bf-lhstrans|block-factors=32,16,64 lhs-transpose-outer-blocks=true lhs-transpose-inner-blocks=true"
  # RHS transpose 끄기 → 순수 [KB][NB][kb][nb] layout (mmt4d 형 아님).
  "matmul|bf-plain|block-factors=32,16,64 rhs-transpose-outer-blocks=false rhs-transpose-inner-blocks=false"
  # positive 2 — generic 전문화 pattern + 이미 transposed 인 RHS.
  "generic-transpose-b|bf8|block-factors=8,8,8"
  # positive 3 — padding 발화 / allow-padding=false 면 no-op.
  "pad-matmul|bf16|block-factors=16,16,16"
  "pad-matmul|nopad|block-factors=16,16,16 allow-padding=false"
  # negative — buffer semantics 는 항상 no-op.
  "negative-memref|bf16|block-factors=16,16,16"
)

fail=0
for run in "${RUNS[@]}"; do
  IFS='|' read -r name combo opts <<<"$run"
  src="$DIR/input/$name.mlir"
  out_my="$DIR/output/output.$combo.$name.mlir"
  out_in="$DIR/output/intree.$combo.$name.mlir"

  if [[ -n "$opts" ]]; then
    "$OPT" "--my-block-pack-matmul=$opts" "$src" -o "$out_my"
    "$OPT" "--linalg-block-pack-matmul=$opts" "$src" -o "$out_in"
  else
    "$OPT" --my-block-pack-matmul "$src" -o "$out_my"
    "$OPT" --linalg-block-pack-matmul "$src" -o "$out_in"
  fi

  if diff -q "$out_my" "$out_in" >/dev/null; then
    echo "[OK   byte-identical] $combo.$name"
  else
    echo "[FAIL divergent     ] $combo.$name"
    fail=1
  fi
done

exit $fail
