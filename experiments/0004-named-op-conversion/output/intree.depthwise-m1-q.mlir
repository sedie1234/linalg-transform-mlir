module {
  func.func @depthwise_conv_q_m1(%arg0: tensor<1x113x113x96xi8>, %arg1: tensor<3x3x96x1xi8>, %arg2: i32, %arg3: i32, %arg4: tensor<1x56x56x96x1xi32>) -> tensor<1x56x56x96x1xi32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<3x3x96x1xi8> into tensor<3x3x96xi8>
    %collapsed_0 = tensor.collapse_shape %arg4 [[0], [1], [2], [3, 4]] : tensor<1x56x56x96x1xi32> into tensor<1x56x56x96xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %collapsed, %arg2, %arg3 : tensor<1x113x113x96xi8>, tensor<3x3x96xi8>, i32, i32) outs(%collapsed_0 : tensor<1x56x56x96xi32>) -> tensor<1x56x56x96xi32>
    %expanded = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [1, 56, 56, 96, 1] : tensor<1x56x56x96xi32> into tensor<1x56x56x96x1xi32>
    return %expanded : tensor<1x56x56x96x1xi32>
  }
}

