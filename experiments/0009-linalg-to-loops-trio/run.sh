#!/usr/bin/env bash
# #0009 linalg-to-loops-trio — 재현 스크립트
#
# 각 입력 × 3 모드를 캡처하고 byte-diff 로 이식 정확성을 검증한다:
#   mode=scf      : output.scf.<name>.mlir      vs intree.scf.<name>.mlir
#                   (--my-linalg-to-loops-trio=mode=scf vs --convert-linalg-to-loops)
#   mode=affine   : output.affine.<name>.mlir   vs intree.affine.<name>.mlir
#                   (… mode=affine vs --convert-linalg-to-affine-loops)
#   mode=parallel : output.parallel.<name>.mlir vs intree.parallel.<name>.mlir
#                   (… mode=parallel vs --convert-linalg-to-parallel-loops)
#
# 세 in-tree pass 모두 anchor 없는 op-agnostic pass 라 module 에 직접
# 올릴 수 있다 (0008 의 InterfacePass 와 달리 -pass-pipeline 불필요).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="$DIR/../../out-of-tree/build/bin/my-mlir-opt"

mkdir -p "$DIR/output"

fail=0
for f in "$DIR"/input/*.mlir; do
  name="$(basename "$f" .mlir)"
  for mode in scf affine parallel; do
    case "$mode" in
      scf)      intree_flag="--convert-linalg-to-loops" ;;
      affine)   intree_flag="--convert-linalg-to-affine-loops" ;;
      parallel) intree_flag="--convert-linalg-to-parallel-loops" ;;
    esac

    "$OPT" --my-linalg-to-loops-trio="mode=$mode" \
        "$f" > "$DIR/output/output.$mode.$name.mlir"
    "$OPT" "$intree_flag" \
        "$f" > "$DIR/output/intree.$mode.$name.mlir"

    if diff -q "$DIR/output/output.$mode.$name.mlir" \
               "$DIR/output/intree.$mode.$name.mlir" >/dev/null; then
      printf '[OK ] byte-identical (mode=%-8s): %s\n' "$mode" "$name"
    else
      printf '[FAIL] divergent      (mode=%-8s): %s\n' "$mode" "$name"
      fail=1
    fi
  done
done

# 추가 엄밀성: in-tree 회귀 테스트 교차 검증 (출력 파일은 남기지 않음).
#   loops.mlir          — convert-linalg-to-loops / -parallel-loops 의 본 테스트
#   affine.mlir         — convert-linalg-to-affine-loops 의 본 테스트
#   parallel-loops.mlir — convert-linalg-to-parallel-loops 의 본 테스트
TESTDIR=/home/hwan/llvm-project/mlir/test/Dialect/Linalg
for pair in "loops.mlir scf" "loops.mlir parallel" "affine.mlir affine" \
            "parallel-loops.mlir parallel"; do
  read -r t mode <<<"$pair"
  case "$mode" in
    scf)      intree_flag="--convert-linalg-to-loops" ;;
    affine)   intree_flag="--convert-linalg-to-affine-loops" ;;
    parallel) intree_flag="--convert-linalg-to-parallel-loops" ;;
  esac
  "$OPT" --allow-unregistered-dialect --split-input-file \
      "$intree_flag" "$TESTDIR/$t" > /tmp/0009.intree 2>&1 || true
  "$OPT" --allow-unregistered-dialect --split-input-file \
      --my-linalg-to-loops-trio="mode=$mode" "$TESTDIR/$t" > /tmp/0009.my 2>&1 || true
  if diff -q /tmp/0009.intree /tmp/0009.my >/dev/null; then
    printf '[OK ] in-tree test byte-identical     : %s (mode=%s)\n' "$t" "$mode"
  else
    printf '[FAIL] in-tree test divergent         : %s (mode=%s)\n' "$t" "$mode"
    fail=1
  fi
done
exit $fail
