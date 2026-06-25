#!/usr/bin/env bash
# T01 — transform dialect 첫 schedule. 같은 payload(@matmul)에 schedule만 바꿔
# -transform-interpreter 로 적용, 같은 바이너리로 다른 IR이 나옴을 보인다.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-tile b-partial c-forall; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A scf.for=$(grep -c 'scf.for ' "$DIR/output/out-a-tile.mlir")  B scf.for=$(grep -c 'scf.for ' "$DIR/output/out-b-partial.mlir")  C scf.forall=$(grep -c 'scf.forall' "$DIR/output/out-c-forall.mlir")"
