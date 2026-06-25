// #0007 positive 입력 3 — generic 이 아닌 tensor op 패턴 3종 발화 케이스.
//
// @pad_unit       : DropPadUnitDims (DropUnitDims.cpp:573-685) —
//                   size==1 ∧ low/high pad==0 인 dim 을 collapse 하고
//                   낮은 rank 로 pad 후 결과를 expand_shape 로 복원.
// @slice_unit     : RankReducedExtractSliceOp (:690-720) —
//                   extract_slice 결과 unit dim 을 rank-reduced slice +
//                   expand_shape 로.
// @insert_unit    : RankReducedInsertSliceOp<tensor::InsertSliceOp>
//                   (:724-757) — source 를 collapse_shape 후 rank-reduced
//                   insert_slice 로.
// (셋 다 ReassociativeReshape 모드 전용 — ExtractInsertSlice 모드의
//  populate (:782-796) 에는 RankReduced*SliceOp 가 없고 DropPadUnitDims 의
//  출력 op 종류가 달라진다.)
func.func @pad_unit(%arg0: tensor<1x16x1xf32>) -> tensor<1x20x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0, 2, 0] high[0, 2, 0] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %cst : f32
  } : tensor<1x16x1xf32> to tensor<1x20x1xf32>
  return %0 : tensor<1x20x1xf32>
}

func.func @slice_unit(%arg0: tensor<8x8xf32>) -> tensor<1x4xf32> {
  %0 = tensor.extract_slice %arg0[2, 0] [1, 4] [1, 1]
      : tensor<8x8xf32> to tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

func.func @insert_unit(%arg0: tensor<1x4xf32>, %arg1: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[2, 0] [1, 4] [1, 1]
      : tensor<1x4xf32> into tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
