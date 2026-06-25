#!/usr/bin/env bash
# T03 — match & handle. Match ops by name, then navigate handles with
# split_handle / get_producer_of_operand / get_parent_op. Each schedule proves
# "which payload op a handle points at" via transform.print and/or a follow-up
# transform (generalize) applied to ONLY that handle.
#   A  print        : match fill + matmul, print each handle (payload unchanged).
#   B  producer     : get_producer_of_operand %matmul[2] -> the linalg.fill;
#                     generalize only that handle => fill becomes linalg.generic
#                     while matmul stays a named op (proves handle = producer).
#   C  split-parent : one match handle -> two generics; split_handle into two
#                     singletons; get_parent_op climbs split#1 up to func.func.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MO="${MLIR_OPT:-/home/hwan/llvm-project/build/bin/mlir-opt}"
for s in a-print b-producer c-split-parent; do
  # full = raw interpreter output (transform.print dumps interleave at the top)
  "$MO" --transform-interpreter "$DIR/input/sched-$s.mlir" > "$DIR/output/full-$s.mlir"
  # print-$s = the IR-printer handle-proof dump (lines before the module/aliases)
  awk '$0 ~ /^(#map|module attributes)/ {exit} {print}' "$DIR/output/full-$s.mlir" > "$DIR/output/print-$s.txt"
  # out-$s = self-contained payload only: slice the real module (incl #map aliases),
  #          inline aliases via local-scope, drop the transform.named_sequence block.
  awk 'p{print; next} /^(#map|module attributes)/{p=1; print}' "$DIR/output/full-$s.mlir" \
    | "$MO" --mlir-print-local-scope \
    | sed '/transform.named_sequence/,/^  }$/d' > "$DIR/output/out-$s.mlir"
done
echo "A: payload unchanged, fill+matmul handles printed"
echo "B: producer handle -> linalg.generic=$(grep -c 'linalg.generic' "$DIR/output/out-b-producer.mlir"), matmul still named=$(grep -c 'linalg.matmul' "$DIR/output/out-b-producer.mlir")"
echo "C: split_handle 2-way + get_parent_op -> func.func (see print-c-split-parent.txt)"
