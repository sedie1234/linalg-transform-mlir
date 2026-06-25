module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %extracted = tensor.extract %arg1[] : tensor<i32>
    %extracted_0 = tensor.extract %arg0[] : tensor<i32>
    cf.br ^bb1(%extracted_0 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %extracted : i32
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = arith.addi %0, %0 : i32
    cf.br ^bb1(%2 : i32)
  ^bb3:  // pred: ^bb1
    %from_elements = tensor.from_elements %0 : tensor<i32>
    return %from_elements : tensor<i32>
  }
}

