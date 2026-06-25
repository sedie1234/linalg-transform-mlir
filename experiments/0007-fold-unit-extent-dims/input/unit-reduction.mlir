// #0007 positive 입력 2 — DropUnitDims + MoveInitOperandsToInput 합주 케이스.
//
// DropUnitDims.cpp:47-82 의 MoveInitOperandsToInput 주석 예제를 실행 가능하게
// 옮긴 것. 4-loop generic 에서 d0(parallel,1)·d2(reduction,1)·d3(reduction,1)
// 이 one-trip 이라 DropUnitDims 가 모두 떨어뜨리면 남는 loop 는 d1(parallel)
// 하나 — all-parallel 이 된다. 그러면 body 가 %out (linalg.fill 결과) 를
// 읽는 elementwise generic 이 되므로 MoveInitOperandsToInput (:83-163) 가
// init 을 ins 로 옮기고 outs 는 새 tensor.empty 로 바꾼다.
func.func @unit_reduction(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1xf32>)
      -> tensor<1x1xf32>
  %2 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d2)>],
       iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0 : tensor<1x?x1x1xf32>) outs(%1 : tensor<1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<1x1xf32>
  return %2 : tensor<1x1xf32>
}
