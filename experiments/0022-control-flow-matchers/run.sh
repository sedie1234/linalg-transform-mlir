#!/usr/bin/env bash
# T12 — 제어흐름·매처. transform.foreach (매치된 op 각각에 변환) + transform.alternatives
# (조건 검사 실패 시 다른 변환 경로). 모두 -transform-interpreter 로 실제 적용.
#   a: match(all linalg) -> foreach { generalize }      (fill+matmul x2 모두 generic)
#   b: match(matmul)     -> foreach { tile }            (각 matmul 독립 scf.for 중첩, fill 무관)
#   c: match(matmul)     -> foreach { tile; generalize } (tile 후 안쪽 matmul generic)
#   d: match(func) -> alternatives { Alt1 실패 -> Alt2 } -> foreach { generalize }
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"

run() {  # $1=slug  $2...=extra mlir-opt flags
  local slug="$1"; shift
  "$MO" --transform-interpreter "$@" "$DIR/input/sched-$slug.mlir" > "$DIR/output/full-$slug.mlir"
  # transform.named_sequence 블록 제거 -> payload 만 남긴 버전.
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$slug.mlir" > "$DIR/output/out-$slug.mlir"
}

run a-foreach-generalize
run b-foreach-tile               --canonicalize
run c-foreach-tile-generalize    --canonicalize
run d-alternatives

echo "a generic=$(grep -c 'linalg.generic' "$DIR/output/out-a-foreach-generalize.mlir") (fill+mm x2=3 기대)"
echo "b scf.for=$(grep -c 'scf.for ' "$DIR/output/out-b-foreach-tile.mlir") (matmul 2개 x 3중첩=6 기대)  fill=$(grep -c 'linalg.fill' "$DIR/output/out-b-foreach-tile.mlir")"
echo "c generic=$(grep -c 'linalg.generic' "$DIR/output/out-c-foreach-tile-generalize.mlir") (inner matmul 2개 generic)  scf.for=$(grep -c 'scf.for ' "$DIR/output/out-c-foreach-tile-generalize.mlir")"
echo "d generic=$(grep -c 'linalg.generic' "$DIR/output/out-d-alternatives.mlir") (matmul 2개만, fill 보존)  fill=$(grep -c 'linalg.fill' "$DIR/output/out-d-alternatives.mlir") (Alt1 폴백 증거)"
