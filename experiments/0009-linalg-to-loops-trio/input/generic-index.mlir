// #0009 입력 2 — linalg.generic, parallel+reduction 혼합 + linalg.index 사용.
//
// 발화 케이스. 두 가지를 관찰:
//   1. iterator_types = [parallel, reduction] → mode=parallel 에서
//      d0 은 scf.parallel, d1 은 그 안의 scf.for 로 **갈라진다**
//      (generateParallelLoopNest 의 재귀 분기, Utils.cpp:408-520).
//   2. body 의 linalg.index 0/1 이 replaceIndexOpsByInductionVariables
//      (Loops.cpp:179-206) 로 실제 loop iv 에 치환된다.
// out[i] += in[i][j] + (i + j) 를 계산.
#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>
func.func @rowsum_with_index(%in: memref<4x8xf32>, %out: memref<4xf32>) {
  linalg.generic {indexing_maps = [#map_in, #map_out],
                  iterator_types = ["parallel", "reduction"]}
      ins(%in : memref<4x8xf32>) outs(%out : memref<4xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %ij = arith.addi %i, %j : index
    %ij_i64 = arith.index_cast %ij : index to i64
    %ij_f = arith.sitofp %ij_i64 : i64 to f32
    %t = arith.addf %a, %ij_f : f32
    %sum = arith.addf %acc, %t : f32
    linalg.yield %sum : f32
  }
  return
}
