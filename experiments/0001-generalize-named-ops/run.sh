#!/usr/bin/env bash
# #0001 generalize-named-ops — 재현 스크립트
#
# 각 입력에 대해:
#   output.<name>.mlir : my-mlir-opt --my-generalize-named-ops      (out-of-tree 재현)
#   intree.<name>.mlir : my-mlir-opt --linalg-generalize-named-ops  (in-tree 원본)
# 을 캡처하고 byte-diff 로 이식 정확성을 검증한다.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="$DIR/../../out-of-tree/build/bin/my-mlir-opt"

mkdir -p "$DIR/output"

fail=0
for f in "$DIR"/input/*.mlir; do
  name="$(basename "$f" .mlir)"
  "$OPT" --my-generalize-named-ops     "$f" > "$DIR/output/output.$name.mlir"
  "$OPT" --linalg-generalize-named-ops "$f" > "$DIR/output/intree.$name.mlir"
  if diff -q "$DIR/output/output.$name.mlir" "$DIR/output/intree.$name.mlir" >/dev/null; then
    echo "[OK ] byte-identical : $name"
  else
    echo "[FAIL] divergent      : $name"
    fail=1
  fi
done
exit $fail
