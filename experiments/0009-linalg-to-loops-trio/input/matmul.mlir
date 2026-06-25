// #0009 입력 1 — named op (linalg.matmul), buffer semantics, dynamic shape.
//
// 발화 케이스. matmul 의 의미 3요소가 loop 로 풀리는 표준 예:
//   iterator_types  = [parallel, parallel, reduction]  → 3중 loop
//   indexing_maps   = (d0,d2)/(d2,d1)/(d0,d1)          → load/store index
//   region(암묵)    = mulf + addf                       → innermost body
// dynamic shape 이라 loop ub 는 memref.dim 으로 나온다 (createLoopRanges
// 가 operand shape 에서 Range 를 만들고, DimOp canonicalization 패턴이
// 같은 greedy 안에서 정리).
func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                  %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%C : memref<?x?xf32>)
  return
}
