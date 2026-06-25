module {
  func.func @pad_unit(%arg0: tensor<1x16x1xf32>) -> tensor<1x20x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 16, 1] [1, 1, 1] : tensor<1x16x1xf32> to tensor<16xf32>
    %padded = tensor.pad %extracted_slice low[2] high[2] {
    ^bb0(%arg1: index):
      tensor.yield %cst : f32
    } : tensor<16xf32> to tensor<20xf32>
    %0 = tensor.empty() : tensor<1x20x1xf32>
    %inserted_slice = tensor.insert_slice %padded into %0[0, 0, 0] [1, 20, 1] [1, 1, 1] : tensor<20xf32> into tensor<1x20x1xf32>
    return %inserted_slice : tensor<1x20x1xf32>
  }
  func.func @slice_unit(%arg0: tensor<8x8xf32>) -> tensor<1x4xf32> {
    %extracted_slice = tensor.extract_slice %arg0[2, 0] [1, 4] [1, 1] : tensor<8x8xf32> to tensor<1x4xf32>
    return %extracted_slice : tensor<1x4xf32>
  }
  func.func @insert_unit(%arg0: tensor<1x4xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %inserted_slice = tensor.insert_slice %arg0 into %arg1[2, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<8x8xf32>
    return %inserted_slice : tensor<8x8xf32>
  }
}

