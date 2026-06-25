// #0002 specialize-generic-ops — elementwise/copy/fill idiom 입력.
// specializeGenericOp (Specialize.cpp:262-309) 의 contraction 이전 분기 4종을
// 모두 발화시킨다:
//   @copy_2d     → isaCopyOpInterface       :264 → linalg.copy
//   @fill_2d     → isaFillOpInterface       :270 → linalg.fill
//   @exp_2d      → isaElemwiseSingleUnary   :276 → linalg.exp
//   @add_2d      → isaElemwiseSingleBinary  :284 → linalg.add
//   @sub_swapped → 〃 + areBinOpsSwapped=true :285 → linalg.sub (operand 교환)
//   @div_2d      → 〃                       :299 → linalg.div
#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// body 가 linalg.yield 단 1개 + maps 모두 identity → copy idiom.
func.func @copy_2d(%src: tensor<8x16xf32>, %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// scalar 입력을 그대로 yield → fill idiom (입력 map 은 rank-0 `()`).
func.func @fill_2d(%cst: f32, %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#scalar2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%cst : f32) outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// body = math.exp 1개 + yield → elemwise single unary idiom.
func.func @exp_2d(%x: tensor<8x16xf32>, %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// body = arith.addf %a, %b (block arg 순서 그대로) → linalg.add.
func.func @add_2d(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>,
                  %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.addf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// body = arith.subf %b, %a (block arg 역순!) → areBinOpsSwapped=true →
// linalg.sub 의 ins 가 (%y, %x) 로 교환되어 생성된다.
func.func @sub_swapped(%x: tensor<8x16xf32>, %y: tensor<8x16xf32>,
                       %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%x, %y : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.subf %in_0, %in : f32
    linalg.yield %s : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// body = arith.divf → linalg.div.
func.func @div_2d(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>,
                  %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic
      {indexing_maps = [#identity2, #identity2, #identity2],
       iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %d = arith.divf %in, %in_0 : f32
    linalg.yield %d : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
