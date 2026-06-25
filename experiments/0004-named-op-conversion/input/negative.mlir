// negative 케이스 3종 — pass 가 발화하지 않아 IR 불변이어야 한다.

// (1) multiplier M=2 (정적이지만 1 이 아님):
//     matchAndReplaceDepthwiseConv :53-54 `kernelTy.getDimSize(3) != 1` bail.
func.func @depthwise_conv_m2(%input: tensor<1x113x113x96xf32>,
                             %kernel: tensor<3x3x96x2xf32>,
                             %init: tensor<1x56x56x96x2xf32>)
    -> tensor<1x56x56x96x2xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
         {dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
         ins(%input, %kernel : tensor<1x113x113x96xf32>, tensor<3x3x96x2xf32>)
         outs(%init : tensor<1x56x56x96x2xf32>) -> tensor<1x56x56x96x2xf32>
  return %0 : tensor<1x56x56x96x2xf32>
}

// (2) multiplier 가 dynamic (`?`): 실제 런타임 값이 1 이어도
//     getDimSize(3) == ShapedType::kDynamic != 1 이라 bail — 판정은 *정적* 기준.
func.func @depthwise_conv_m_dyn(%input: tensor<1x113x113x96xf32>,
                                %kernel: tensor<3x3x96x?xf32>,
                                %init: tensor<1x56x56x96x?xf32>)
    -> tensor<1x56x56x96x?xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
         {dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
         ins(%input, %kernel : tensor<1x113x113x96xf32>, tensor<3x3x96x?xf32>)
         outs(%init : tensor<1x56x56x96x?xf32>) -> tensor<1x56x56x96x?xf32>
  return %0 : tensor<1x56x56x96x?xf32>
}

// (3) memref semantics: M=1 이어도 hasPureTensorSemantics() :42 에서 bail —
//     변환이 만드는 collapse_shape/expand_shape 가 tensor op 이기 때문.
func.func @depthwise_conv_memref(%input: memref<1x113x113x96xf32>,
                                 %kernel: memref<3x3x96x1xf32>,
                                 %init: memref<1x56x56x96x1xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwcm
    {dilations = dense<1> : tensor<2xi64>,
     strides = dense<2> : tensor<2xi64>}
    ins(%input, %kernel : memref<1x113x113x96xf32>, memref<3x3x96x1xf32>)
    outs(%init : memref<1x56x56x96x1xf32>)
  return
}
