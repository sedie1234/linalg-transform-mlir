#!/usr/bin/env bash
# T07 — vectorize. linalg -> vector dialect. 같은 static payload 에 vectorizer 만 바꿔
# -transform-interpreter 로 적용.
#   a-vectorize         : transform.structured.vectorize (matmul) -> vector.multi_reduction
#   b-children-patterns : vectorize_children_and_apply_patterns (func) -> vector.contract
#   c-elementwise       : transform.structured.vectorize (generic) -> arith.addf <vector>
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-vectorize b-children-patterns c-elementwise; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "A multi_reduction=$(grep -c 'vector.multi_reduction' "$DIR/output/out-a-vectorize.mlir")  B contract=$(grep -c 'vector.contract' "$DIR/output/out-b-children-patterns.mlir")  C transfer_read=$(grep -c 'vector.transfer_read' "$DIR/output/out-c-elementwise.mlir")"
