// #0001 입력 1 (positive, tensor semantics)
// named op 3종이 모두 LinalgGeneralizationPattern 에 의해 linalg.generic 으로
// 바뀐다.  hasPureTensorSemantics == true 이므로 generalizeNamedOp 의
// resultTypes = TypeRange(outputs) 분기 (Generalization.cpp:62-64) 를 탄다.
func.func @tensor_named(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>,
                        %C: tensor<4x16xf32>,
                        %x: tensor<4x16xf32>, %y: tensor<4x16xf32>)
    -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<16x4xf32>) {
  // matmul: iterator [parallel, parallel, reduction], payload mulf+addf
  %0 = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
                     outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
  // add: elementwise, iterator [parallel, parallel], payload addf
  %1 = linalg.add ins(%x, %y : tensor<4x16xf32>, tensor<4x16xf32>)
                  outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
  // transpose: permutation 이 indexing_maps 로 변환되는 케이스
  %init = tensor.empty() : tensor<16x4xf32>
  %2 = linalg.transpose ins(%0 : tensor<4x16xf32>)
                        outs(%init : tensor<16x4xf32>) permutation = [1, 0]
  return %1, %0, %2 : tensor<4x16xf32>, tensor<4x16xf32>, tensor<16x4xf32>
}
