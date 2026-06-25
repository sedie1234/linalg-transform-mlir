#!/usr/bin/env bash
# T08 — representation transforms. 같은 바이너리로 4가지 표현 변환을 schedule 로 적용:
#   generalize  : linalg.matmul   -> linalg.generic  (named -> generic)
#   specialize  : copy generic    -> linalg.copy     (generic -> named, 역방향)
#   interchange : generic iter [0,1] -> [1,0]         (indexing_maps 재작성, generic 전용)
#   decompose   : linalg.softmax  -> fill/max/exp/sum/div generic 시퀀스
# full-*.mlir = transform script 포함 변환 결과, out-*.mlir = payload 만 남긴 버전.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in generalize specialize interchange decompose; do
  "$MO" --transform-interpreter "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  sed '/transform.named_sequence/,/^  }$/d' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "generalize: payload generic=$(grep -c 'linalg.generic' "$DIR/output/out-generalize.mlir")  (matmul 소멸)"
echo "specialize: payload copy=$(grep -c 'linalg.copy' "$DIR/output/out-specialize.mlir")  (generic 소멸)"
echo "interchange: transposed map=$(grep -c '(d0, d1) -> (d1, d0)' "$DIR/output/out-interchange.mlir")"
echo "decompose : payload generic=$(grep -c 'linalg.generic' "$DIR/output/out-decompose.mlir")  fill=$(grep -c 'linalg.fill' "$DIR/output/out-decompose.mlir")  (softmax 소멸)"
