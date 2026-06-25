#!/usr/bin/env bash
# T02 — 같은 변환(generalize)을 두 방식으로: 고정 패스 vs transform script.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
"$MO" -linalg-generalize-named-ops "$DIR/input/payload.mlir" > "$DIR/output/out-A-pass.mlir"
"$MO" -transform-interpreter "$DIR/input/sched-generalize-matmul.mlir" | sed '/transform.named_sequence/,/^  }$/d' > "$DIR/output/out-B-transform.mlir"
"$MO" -transform-interpreter "$DIR/input/sched-compose.mlir" | sed '/transform.named_sequence/,/^  }$/d' > "$DIR/output/out-C-compose.mlir"
echo "A: $(grep -c 'linalg.generic' "$DIR/output/out-A-pass.mlir") generic / B: matmul만 / C: tile+generalize 조합"
