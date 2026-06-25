// 발화 케이스 (a): rank-0 tensor operand.
// %scalar 의 indexing map 은 (d0) -> () — result 0개.
// AffineMap::isConstant() 는 all_of(empty) == true 이므로 scalar 로 판정,
// body 선두에 tensor.extract %scalar[] (인덱스 0개) 로 inline 된다.
#map_scalar = affine_map<(d0) -> ()>
#map_id     = affine_map<(d0) -> (d0)>

func.func @inline_rank0(%scalar: tensor<f32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %res = linalg.generic
      {indexing_maps = [#map_scalar, #map_id, #map_id],
       iterator_types = ["parallel"]}
      ins(%scalar, %x : tensor<f32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%s: f32, %xi: f32, %o: f32):
    %sum = arith.addf %s, %xi : f32
    linalg.yield %sum : f32
  } -> tensor<8xf32>
  return %res : tensor<8xf32>
}
