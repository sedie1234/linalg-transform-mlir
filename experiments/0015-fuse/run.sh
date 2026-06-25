#!/usr/bin/env bash
# T05 — fuse. producer(elementwise) -> consumer 를 한 loop nest 안으로 끌어들인다.
#   A: transform.structured.fuse           (tile+fuse, matmul root, scf.for)
#   B: tile_using_forall + fuse_into_containing_op  (scf.forall container)
#   C: transform.structured.fuse           (elemwise->elemwise, scf.for)
# 같은 바이너리(-transform-interpreter)로 schedule 만 바꿔 적용한다.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-fuse b-fuse-forall c-fuse-elemwise; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A scf.for=$(grep -c 'scf.for ' "$DIR/output/out-a-fuse.mlir")  B scf.forall=$(grep -c 'scf.forall' "$DIR/output/out-b-fuse-forall.mlir")  C scf.for=$(grep -c 'scf.for ' "$DIR/output/out-c-fuse-elemwise.mlir")"
