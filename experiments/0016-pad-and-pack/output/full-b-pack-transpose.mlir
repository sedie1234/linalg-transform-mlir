#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @matmul(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = tensor.empty() : tensor<8x4x4x8xf32>
    %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %0 : tensor<32x32xf32> -> tensor<8x4x4x8xf32>
    %1 = tensor.empty() : tensor<4x4x8x8xf32>
    %2 = tensor.empty() : tensor<4x4x8x8xf32>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %2 : tensor<32x32xf32> -> tensor<4x4x8x8xf32>
    %3 = tensor.empty() : tensor<8x4x4x8xf32>
    %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %3 : tensor<32x32xf32> -> tensor<8x4x4x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<8x4x4x8xf32>, tensor<4x4x8x8xf32>) outs(%pack_1 : tensor<8x4x4x8xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<8x4x4x8xf32>
    %unpack = tensor.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %arg2 : tensor<8x4x4x8xf32> -> tensor<32x32xf32>
    return %unpack : tensor<32x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.pack %0 packed_sizes = [4, 8, 8] : (!transform.any_op) -> !transform.op<"linalg.generic">
      %2 = transform.get_producer_of_operand %1[1] : (!transform.op<"linalg.generic">) -> !transform.op<"tensor.pack">
      %packed_op, %pack_op, %un_pack_op = transform.structured.pack_transpose %2 with_compute_op(%1) outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.op<"tensor.pack">, !transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.op<"tensor.pack">, !transform.any_op)
      transform.yield 
    }
  }
}

