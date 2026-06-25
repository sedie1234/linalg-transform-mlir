module {
  func.func @depthwise_conv_m1(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x96x1xf32>, %arg2: tensor<1x56x56x96x1xf32>) -> tensor<1x56x56x96x1xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<3x3x96x1xf32> into tensor<3x3x96xf32>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2], [3, 4]] : tensor<1x56x56x96x1xf32> into tensor<1x56x56x96xf32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwc {_someattr, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %collapsed : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%collapsed_0 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    %expanded = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [1, 56, 56, 96, 1] : tensor<1x56x56x96xf32> into tensor<1x56x56x96x1xf32>
    return %expanded : tensor<1x56x56x96x1xf32>
  }
  func.func @depthwise_conv_m1_dyn(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x1xf32>, %arg2: tensor<?x?x?x?x1xf32>) -> tensor<?x?x?x?x1xf32> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<?x?x?x1xf32> into tensor<?x?x?xf32>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2], [3, 4]] : tensor<?x?x?x?x1xf32> into tensor<?x?x?x?xf32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %collapsed : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%collapsed_0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %dim = tensor.dim %0, %c0 : tensor<?x?x?x?xf32>
    %dim_1 = tensor.dim %0, %c1 : tensor<?x?x?x?xf32>
    %dim_2 = tensor.dim %0, %c2 : tensor<?x?x?x?xf32>
    %dim_3 = tensor.dim %0, %c3 : tensor<?x?x?x?xf32>
    %expanded = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [%dim, %dim_1, %dim_2, %dim_3, 1] : tensor<?x?x?x?xf32> into tensor<?x?x?x?x1xf32>
    return %expanded : tensor<?x?x?x?x1xf32>
  }
}

