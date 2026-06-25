// sched-C: interchange — generic 의 iterator 순서 [0,1] → [1,0].
// indexing_maps 가 (d0,d1)->(d0,d1) 에서 (d0,d1)->(d1,d0) 로 재작성된다.
// interchange 는 generic 전용 (named op 에 적용하면 wrong-op-kind 에러).
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @elemwise_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {
      indexing_maps = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1 = math.exp %in : f32
        linalg.yield %1 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.interchange %g iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
