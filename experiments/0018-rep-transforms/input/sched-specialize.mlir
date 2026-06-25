// sched-B: specialize — trivial-copy linalg.generic → linalg.copy (generic → named).
// generalize 의 역방향: 패턴이 알려진 named op 와 일치하면 특수화한다.
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @copy_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {
      indexing_maps = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %s = transform.structured.specialize %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
