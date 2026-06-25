#!/usr/bin/env bash
# #0007 fold-unit-extent-dims — 재현 스크립트
#
# 각 입력에 대해 두 모드를 캡처하고 byte-diff 로 이식 정확성을 검증한다:
#   [기본 모드: ReassociativeReshape]
#     output.<name>.mlir : my-mlir-opt --my-fold-unit-extent-dims
#     intree.<name>.mlir : my-mlir-opt --linalg-fold-unit-extent-dims
#   [옵션 모드: use-rank-reducing-slices=true → ExtractInsertSlice]
#     output.slices.<name>.mlir / intree.slices.<name>.mlir
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="$DIR/../../out-of-tree/build/bin/my-mlir-opt"

mkdir -p "$DIR/output"

fail=0
for f in "$DIR"/input/*.mlir; do
  name="$(basename "$f" .mlir)"

  # 기본 모드 (ReassociativeReshape)
  "$OPT" --my-fold-unit-extent-dims     "$f" > "$DIR/output/output.$name.mlir"
  "$OPT" --linalg-fold-unit-extent-dims "$f" > "$DIR/output/intree.$name.mlir"
  if diff -q "$DIR/output/output.$name.mlir" "$DIR/output/intree.$name.mlir" >/dev/null; then
    echo "[OK ] byte-identical (reshape mode) : $name"
  else
    echo "[FAIL] divergent      (reshape mode) : $name"
    fail=1
  fi

  # 옵션 모드 (ExtractInsertSlice)
  "$OPT" "--my-fold-unit-extent-dims=use-rank-reducing-slices=true" \
      "$f" > "$DIR/output/output.slices.$name.mlir"
  "$OPT" "--linalg-fold-unit-extent-dims=use-rank-reducing-slices=true" \
      "$f" > "$DIR/output/intree.slices.$name.mlir"
  if diff -q "$DIR/output/output.slices.$name.mlir" "$DIR/output/intree.slices.$name.mlir" >/dev/null; then
    echo "[OK ] byte-identical (slices mode)  : $name"
  else
    echo "[FAIL] divergent      (slices mode)  : $name"
    fail=1
  fi
done
exit $fail
