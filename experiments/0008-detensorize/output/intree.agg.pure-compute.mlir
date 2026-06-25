module {
  func.func @detensorable_but_no_cf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %extracted = tensor.extract %arg1[] : tensor<f32>
    %extracted_0 = tensor.extract %arg0[] : tensor<f32>
    %0 = arith.addf %extracted_0, %extracted : f32
    %from_elements = tensor.from_elements %0 : tensor<f32>
    return %from_elements : tensor<f32>
  }
}

