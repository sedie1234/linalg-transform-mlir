#!/usr/bin/env bash
# #0008 detensorize — 재현 스크립트
#
# 각 입력에 대해 두 모드를 캡처하고 byte-diff 로 이식 정확성을 검증한다:
#   [기본 모드: ControlFlowDetectionModel]
#     output.<name>.mlir : my-mlir-opt func.func(my-detensorize)
#     intree.<name>.mlir : my-mlir-opt func.func(linalg-detensorize)
#   [aggressive-mode: AggressiveDetensoringModel]
#     output.agg.<name>.mlir / intree.agg.<name>.mlir
#
# 주의: linalg-detensorize 는 InterfacePass<FunctionOpInterface> 라서
# module 에 직접 못 올린다 — 반드시 -pass-pipeline 으로 func.func 안에
# anchoring 해야 한다 (`--linalg-detensorize` 단독 호출은
# "unable to schedule pass" 에러).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="$DIR/../../out-of-tree/build/bin/my-mlir-opt"

mkdir -p "$DIR/output"

fail=0
for f in "$DIR"/input/*.mlir; do
  name="$(basename "$f" .mlir)"

  # 기본 모드 (ControlFlowDetectionModel)
  "$OPT" -pass-pipeline="builtin.module(func.func(my-detensorize))" \
      "$f" > "$DIR/output/output.$name.mlir"
  "$OPT" -pass-pipeline="builtin.module(func.func(linalg-detensorize))" \
      "$f" > "$DIR/output/intree.$name.mlir"
  if diff -q "$DIR/output/output.$name.mlir" "$DIR/output/intree.$name.mlir" >/dev/null; then
    echo "[OK ] byte-identical (default mode)    : $name"
  else
    echo "[FAIL] divergent      (default mode)    : $name"
    fail=1
  fi

  # aggressive-mode (AggressiveDetensoringModel)
  "$OPT" -pass-pipeline="builtin.module(func.func(my-detensorize{aggressive-mode}))" \
      "$f" > "$DIR/output/output.agg.$name.mlir"
  "$OPT" -pass-pipeline="builtin.module(func.func(linalg-detensorize{aggressive-mode}))" \
      "$f" > "$DIR/output/intree.agg.$name.mlir"
  if diff -q "$DIR/output/output.agg.$name.mlir" "$DIR/output/intree.agg.$name.mlir" >/dev/null; then
    echo "[OK ] byte-identical (aggressive mode) : $name"
  else
    echo "[FAIL] divergent      (aggressive mode) : $name"
    fail=1
  fi
done

# 추가 엄밀성: in-tree 회귀 테스트 8종 전체 교차 검증 (출력 파일은 남기지 않음)
for t in /home/hwan/llvm-project/mlir/test/Dialect/Linalg/detensorize_*.mlir; do
  n="$(basename "$t" .mlir)"
  for mode in "" "{aggressive-mode}"; do
    "$OPT" --allow-unregistered-dialect --split-input-file \
        -pass-pipeline="builtin.module(func.func(linalg-detensorize$mode))" "$t" > /tmp/0008.intree 2>&1 || true
    "$OPT" --allow-unregistered-dialect --split-input-file \
        -pass-pipeline="builtin.module(func.func(my-detensorize$mode))" "$t" > /tmp/0008.my 2>&1 || true
    if diff -q /tmp/0008.intree /tmp/0008.my >/dev/null; then
      echo "[OK ] in-tree test byte-identical     : $n ${mode:-default}"
    else
      echo "[FAIL] in-tree test divergent         : $n ${mode:-default}"
      fail=1
    fi
  done
done
exit $fail
