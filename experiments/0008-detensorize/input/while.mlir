// 0-d tensor 로 표현된 while loop (in-tree test detensorize_while.mlir 기반).
// loop 의 조건(cmpi)과 본문(addi)이 모두 0-d tensor 위의 linalg.generic 으로
// 표현되어 있고, 그 값들이 cf.br/cf.cond_br 의 operand 로 block 경계를 넘는다.
// → ControlFlowDetectionModel(기본 모드)이 cond_br 의 operand 에서 출발해
//   use-def chain 으로 "detensoring component" 를 발견 → 두 generic 모두 발화.
#map0 = affine_map<() -> ()>

#attrs = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = []
}

func.func @main(%farg0: tensor<i32>, %farg1: tensor<i32>) -> tensor<i32> attributes {} {
  cf.br ^bb1(%farg0 : tensor<i32>)

^bb1(%0: tensor<i32>):  // 2 preds: ^bb0, ^bb2
  %1 = tensor.empty() : tensor<i1>
  %2 = linalg.generic #attrs
    ins(%0, %farg1 : tensor<i32>, tensor<i32>)
    outs(%1 : tensor<i1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %8 : i1
  } -> tensor<i1>
  %3 = tensor.extract %2[] : tensor<i1>
  cf.cond_br %3, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)

^bb2(%4: tensor<i32>):  // pred: ^bb1
  %5 = tensor.empty() : tensor<i32>
  %6 = linalg.generic #attrs
    ins(%4, %4 : tensor<i32>, tensor<i32>)
    outs(%5 : tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
      %8 = arith.addi %arg0, %arg1 : i32
      linalg.yield %8 : i32
  } -> tensor<i32>
  cf.br ^bb1(%6 : tensor<i32>)

^bb3(%7: tensor<i32>):  // pred: ^bb1
  return %7 : tensor<i32>
}
