// T08 payload — representation transforms (generalize/specialize/interchange/decompose).
// 각 함수가 한 변환의 대상이 된다. transform script 없이 IR 검증만 할 때 쓰는 baseline.

#id2  = affine_map<(d0, d1) -> (d0, d1)>

// generalize 대상: named op (matmul) → linalg.generic
func.func @mm(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
                     outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// specialize 대상: trivial-copy generic → linalg.copy
func.func @copy_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#id2, #id2],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// interchange 대상: elementwise generic, iterator [d0,d1] → [d1,d0]
func.func @elemwise_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#id2, #id2],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp %in : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// decompose 대상: linalg.softmax → fill/max/exp/sum/div generic 시퀀스
func.func @sm(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}
