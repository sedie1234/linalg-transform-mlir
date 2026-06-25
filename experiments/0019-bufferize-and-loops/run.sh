#!/usr/bin/env bash
# T09 — bufferize & to-loops. Three schedules over a matmul:
#   A bufferize_to_allocation (tensor -> explicit memref.alloc, no loops)
#   B convert_to_loops        (memref matmul -> scf.for nest, scalar body)
#   C A + one_shot_bufferize + B (full tensor->memref->loops pipeline)
# full-*.mlir keeps the transform script; out-*.mlir strips it (payload only).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-buffer b-loops c-buffer-then-loops; do
  "$MO" -transform-interpreter "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A memref.alloc=$(grep -c 'memref.alloc' "$DIR/output/out-a-buffer.mlir")  B scf.for=$(grep -c 'scf.for ' "$DIR/output/out-b-loops.mlir")  C scf.for=$(grep -c 'scf.for ' "$DIR/output/out-c-buffer-then-loops.mlir") memref.alloc=$(grep -c 'memref.alloc' "$DIR/output/out-c-buffer-then-loops.mlir")"
