#!/usr/bin/env bash
# T11 — schedule 변형 sweep (autotuning 스케치). 같은 payload(@matmul)·같은 바이너리,
# transform script의 tile size만 바꿔 다른 IR이 나옴을 보인다.
#   a-8       : tile_using_for [8,8,8]    -> scf.for x3, step 8
#   b-32      : tile_using_for [32,32,32] -> scf.for x3, step 32 (구조 동일, step만 변)
#   c-64x64x0 : tile_using_for [64,64,0]  -> K 미타일 -> scf.for x1 (loop 2개로 축소)
#   d-forall  : tile_using_forall [64,32] -> scf.forall (2,2) 병렬 (loop construct 자체가 다름)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-8 b-32 c-64x64x0 d-forall; do
  "$MO" --transform-interpreter --canonicalize "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "a-8 scf.for=$(grep -c 'scf.for ' "$DIR/output/out-a-8.mlir")  b-32 scf.for=$(grep -c 'scf.for ' "$DIR/output/out-b-32.mlir")  c-64x64x0 scf.for=$(grep -c 'scf.for ' "$DIR/output/out-c-64x64x0.mlir")  d-forall scf.forall=$(grep -c 'scf.forall ' "$DIR/output/out-d-forall.mlir")"
