#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func @broadcast_add(%arg0: tensor<1x5xf32>, %arg1: tensor<5x1xf32>) -> tensor<5x5xf32> {
    %0 = tensor.empty() : tensor<5x5xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 5] [1, 1] : tensor<1x5xf32> to tensor<5xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [5, 1] [1, 1] : tensor<5x1xf32> to tensor<5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<5xf32>, tensor<5xf32>) outs(%0 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<5x5xf32>
    return %1 : tensor<5x5xf32>
  }
  func.func @drop_unit_loop_with_index(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %0 = tensor.empty() : tensor<1x8xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<8xf32>) outs(%1 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = linalg.index 0 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.addf %in, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<8xf32>
    %inserted_slice = tensor.insert_slice %2 into %0[0, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<1x8xf32>
    return %inserted_slice : tensor<1x8xf32>
  }
}

