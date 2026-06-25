// Sched C — elementwise linalg.generic 를 vectorize.
// matmul(contraction)과 달리 contract 없이 순수 elementwise:
//   vector.transfer_read(A), transfer_read(B), arith.addf <vector>, transfer_write.
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @add_elemwise(%A: tensor<8x16xf32>, %B: tensor<8x16xf32>,
                          %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]
         } ins(%A, %B : tensor<8x16xf32>, tensor<8x16xf32>)
           outs(%C : tensor<8x16xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %s = arith.addf %a, %b : f32
        linalg.yield %s : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op
    transform.yield
  }
}
