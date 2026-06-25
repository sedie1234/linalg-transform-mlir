#!/usr/bin/env bash
# T06 — pad & pack. 같은 payload(matmul)에 pack / pack+pack_transpose /
# pack_greedily+lower_pack 세 schedule을 -transform-interpreter 로 적용.
# tensor.pack/unpack 생성, transpose 레이아웃, pack→pad+expand_shape+transpose
# lowering 을 IR로 캡처한다. (0010 block-pack 효과를 schedule IR로 명시한 버전.)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-pack b-pack-transpose c-greedily-lower; do
  "$MO" --transform-interpreter "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  # transform script(= module attributes {transform.with_named_sequence} {...})
  # 블록을 통째로 제거해 payload만 남긴다. mlir-opt가 결과를 outer module로
  # 감싸므로 단순 line-range sed로는 안 떨어진다 → brace 카운팅으로 정확히 도려낸다.
  awk '
    /module attributes \{transform.with_named_sequence\}/ { skip=1; depth=0 }
    skip {
      n=gsub(/\{/,"{"); depth+=n
      m=gsub(/\}/,"}"); depth-=m
      if (depth<=0) { skip=0 }
      next
    }
    { print }
  ' "$DIR/output/full-$s.mlir" > "$DIR/output/out-$s.mlir"
done
echo "a-pack   pack=$(grep -c 'tensor.pack ' "$DIR/output/out-a-pack.mlir")  unpack=$(grep -c 'tensor.unpack ' "$DIR/output/out-a-pack.mlir")"
echo "b-transp pack=$(grep -c 'tensor.pack ' "$DIR/output/out-b-pack-transpose.mlir")  (B pack has outer_dims_perm=[1,0])"
echo "c-lower  pad=$(grep -c 'tensor.pad ' "$DIR/output/out-c-greedily-lower.mlir")  expand_shape=$(grep -c 'tensor.expand_shape' "$DIR/output/out-c-greedily-lower.mlir")  transpose=$(grep -c 'linalg.transpose' "$DIR/output/out-c-greedily-lower.mlir")"
