#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @unit_reduction(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %dim = tensor.dim %arg0, %c1 : tensor<1x?x1x1xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, %dim, 1, 1] [1, 1, 1, 1] : tensor<1x?x1x1xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<f32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<?xf32>, tensor<f32>) outs(%2 : tensor<f32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0] [1, 1] [1, 1] : tensor<f32> into tensor<1x1xf32>
    return %inserted_slice : tensor<1x1xf32>
  }
}

