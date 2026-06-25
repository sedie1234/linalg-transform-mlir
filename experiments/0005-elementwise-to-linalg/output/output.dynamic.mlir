#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @dynamic_elementwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi1>) {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.addf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %0, %c1 : tensor<?x?xf32>
    %1 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xi1>) {
    ^bb0(%in: f32, %in_1: f32, %out: i1):
      %3 = arith.cmpf olt, %in, %in_1 : f32
      linalg.yield %3 : i1
    } -> tensor<?x?xi1>
    return %0, %2 : tensor<?x?xf32>, tensor<?x?xi1>
  }
}

