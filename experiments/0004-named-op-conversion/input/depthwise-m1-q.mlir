// 발화 케이스 (양자화): depthwise_conv_2d_nhwc_hwcm_q (DPS input 4개 —
// input, kernel, input zero-point iZp, kernel zero-point kZp).
// M=1 → SimplifyDepthwiseConvQOp 가 발화해 depthwise_conv_2d_nhwc_hwc_q 로.
// iZp/kZp 스칼라 operand 는 collapse 대상이 아니라 그대로 전달된다
// (matchAndReplaceDepthwiseConv 의 ValueRange{input, collapsedKernel, iZp, kZp}).
func.func @depthwise_conv_q_m1(%input: tensor<1x113x113x96xi8>,
                               %kernel: tensor<3x3x96x1xi8>,
                               %iZp: i32, %kZp: i32,
                               %init: tensor<1x56x56x96x1xi32>)
    -> tensor<1x56x56x96x1xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q
         {dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
         ins(%input, %kernel, %iZp, %kZp
             : tensor<1x113x113x96xi8>, tensor<3x3x96x1xi8>, i32, i32)
         outs(%init : tensor<1x56x56x96x1xi32>) -> tensor<1x56x56x96x1xi32>
  return %0 : tensor<1x56x56x96x1xi32>
}
