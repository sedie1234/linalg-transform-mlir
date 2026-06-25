module {
  func.func @depthwise_conv_m2(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x96x2xf32>, %arg2: tensor<1x56x56x96x2xf32>) -> tensor<1x56x56x96x2xf32> {
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x113x113x96xf32>, tensor<3x3x96x2xf32>) outs(%arg2 : tensor<1x56x56x96x2xf32>) -> tensor<1x56x56x96x2xf32>
    return %0 : tensor<1x56x56x96x2xf32>
  }
  func.func @depthwise_conv_m_dyn(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x96x?xf32>, %arg2: tensor<1x56x56x96x?xf32>) -> tensor<1x56x56x96x?xf32> {
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x113x113x96xf32>, tensor<3x3x96x?xf32>) outs(%arg2 : tensor<1x56x56x96x?xf32>) -> tensor<1x56x56x96x?xf32>
    return %0 : tensor<1x56x56x96x?xf32>
  }
  func.func @depthwise_conv_memref(%arg0: memref<1x113x113x96xf32>, %arg1: memref<3x3x96x1xf32>, %arg2: memref<1x56x56x96x1xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x113x113x96xf32>, memref<3x3x96x1xf32>) outs(%arg2 : memref<1x56x56x96x1xf32>)
    return
  }
}

