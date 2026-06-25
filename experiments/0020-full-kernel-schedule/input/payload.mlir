// T10 payload — fc + bias + relu (tutorial Ch1 형태). 3 linalg op.
//   %matmul = linalg.matmul                 (fc)
//   %biased = elemwise_binary<add>          (bias)
//   %relued = elemwise_binary<max_signed>   (relu, max with 0)
func.func @fc_relu(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>,
                   %bias: tensor<128x128xf32>, %output: tensor<128x128xf32>)
                   -> tensor<128x128xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<128x128xf32>, tensor<128x128xf32>)
                          outs(%output: tensor<128x128xf32>) -> tensor<128x128xf32>

  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%output : tensor<128x128xf32>) -> tensor<128x128xf32>

  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<128x128xf32>, f32)
    outs(%output : tensor<128x128xf32>) -> tensor<128x128xf32>
  func.return %relued : tensor<128x128xf32>
}
