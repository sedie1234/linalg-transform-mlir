#!/usr/bin/env bash
# T04 (0014-tile) — 같은 linalg.matmul payload(128x256 x 256x64)에 tile schedule만
# 바꿔 -transform-interpreter 로 적용. tile_using_for(scf.for, 순차) vs
# tile_using_forall(scf.forall, 병렬; num_threads vs tile_sizes) 전후 IR 비교.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-for b-forall-numthreads c-forall-tilesizes; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  # transform.named_sequence 블록 제거 → payload만 남긴 변환 결과
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A(for) scf.for=$(grep -c 'scf.for ' "$DIR/output/out-a-for.mlir")  B(forall/num_threads) scf.forall=$(grep -c 'scf.forall ' "$DIR/output/out-b-forall-numthreads.mlir")  C(forall/tile_sizes) scf.forall=$(grep -c 'scf.forall ' "$DIR/output/out-c-forall-tilesizes.mlir")"
echo "B vs C payload diff: $(diff -q "$DIR/output/out-b-forall-numthreads.mlir" "$DIR/output/out-c-forall-tilesizes.mlir" >/dev/null 2>&1 && echo IDENTICAL || echo DIFFER)"
