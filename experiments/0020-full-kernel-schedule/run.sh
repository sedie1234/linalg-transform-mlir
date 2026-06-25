#!/usr/bin/env bash
# T10 — full kernel schedule. fc+bias+relu payload (matmul → elemwise add →
# elemwise max 0). One named_sequence chains match → tile → fuse → vectorize,
# incrementally adding stages (A tile, B +fuse, C +vectorize). Same binary,
# schedule grows step by step; final IR captured per stage.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-tile b-fuse c-vectorize; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  # strip the transform.named_sequence block, leaving payload-only IR
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A forall=$(grep -c 'scf.forall ' "$DIR/output/out-a-tile.mlir")  matmul=$(grep -c 'linalg.matmul' "$DIR/output/out-a-tile.mlir")"
echo "B forall=$(grep -c 'scf.forall ' "$DIR/output/out-b-fuse.mlir")  matmul-in-loop=$(grep -c 'linalg.matmul' "$DIR/output/out-b-fuse.mlir")"
echo "C forall=$(grep -c 'scf.forall ' "$DIR/output/out-c-vectorize.mlir")  vector.contract=$(grep -c 'vector.contract' "$DIR/output/out-c-vectorize.mlir")  linalg-left=$(grep -c 'linalg\.' "$DIR/output/out-c-vectorize.mlir")"
