module {
  func.func @pad_unit(%arg0: tensor<1x16x1xf32>) -> tensor<1x20x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x16x1xf32> into tensor<16xf32>
    %padded = tensor.pad %collapsed low[2] high[2] {
    ^bb0(%arg1: index):
      tensor.yield %cst : f32
    } : tensor<16xf32> to tensor<20xf32>
    %expanded = tensor.expand_shape %padded [[0, 1, 2]] output_shape [1, 20, 1] : tensor<20xf32> into tensor<1x20x1xf32>
    return %expanded : tensor<1x20x1xf32>
  }
  func.func @slice_unit(%arg0: tensor<8x8xf32>) -> tensor<1x4xf32> {
    %extracted_slice = tensor.extract_slice %arg0[2, 0] [1, 4] [1, 1] : tensor<8x8xf32> to tensor<4xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    return %expanded : tensor<1x4xf32>
  }
  func.func @insert_unit(%arg0: tensor<1x4xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x4xf32> into tensor<4xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %arg1[2, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<8x8xf32>
    return %inserted_slice : tensor<8x8xf32>
  }
}

