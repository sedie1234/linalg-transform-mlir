// 발화 케이스 (비양자화): depthwise_conv_2d_nhwc_hwcm, kernel 의 마지막 차원
// (channel multiplier M) 이 *정적으로* 1 → SimplifyDepthwiseConvOp 가 발화해
//   tensor.collapse_shape(kernel [[0],[1],[2,3]])
//   tensor.collapse_shape(init   [[0],[1],[2],[3,4]])
//   linalg.depthwise_conv_2d_nhwc_hwc
//   tensor.expand_shape(result   [[0],[1],[2],[3,4]])
// 로 바뀐다. discardable attr `_someattr` 는 getPrunedAttributeList 로 보존.
func.func @depthwise_conv_m1(%input: tensor<1x113x113x96xf32>,
                             %kernel: tensor<3x3x96x1xf32>,
                             %init: tensor<1x56x56x96x1xf32>)
    -> tensor<1x56x56x96x1xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
         {_someattr, dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
         ins(%input, %kernel : tensor<1x113x113x96xf32>, tensor<3x3x96x1xf32>)
         outs(%init : tensor<1x56x56x96x1xf32>) -> tensor<1x56x56x96x1xf32>
  return %0 : tensor<1x56x56x96x1xf32>
}

// 발화 케이스 (dynamic 이지만 M 차원만 정적 1): in-tree 테스트와 동일 모양.
// multiplier 판정은 kernelTy.getDimSize(3) == 1 만 보므로 나머지 차원이
// dynamic 이어도 발화한다.
func.func @depthwise_conv_m1_dyn(%input: tensor<?x?x?x?xf32>,
                                 %kernel: tensor<?x?x?x1xf32>,
                                 %init: tensor<?x?x?x?x1xf32>)
    -> tensor<?x?x?x?x1xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
         {dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
         ins(%input, %kernel : tensor<?x?x?x?xf32>, tensor<?x?x?x1xf32>)
         outs(%init : tensor<?x?x?x?x1xf32>) -> tensor<?x?x?x?x1xf32>
  return %0 : tensor<?x?x?x?x1xf32>
}
