// C — one match handle covering BOTH linalg.generic ops, split_handle into two
//     singleton handles, then get_parent_op to climb from a leaf op up to its
//     enclosing func.func. Prints prove the partition + the climb.
//     Follow-up proof: fuse %mul (2nd generic) into %add's loop is overkill;
//     instead we generalize-by-interchange the first split handle's parent
//     is shown via print. The distinct prints show split selected distinct ops.
module attributes {transform.with_named_sequence} {
  func.func @two_gen(%a: tensor<32xf32>, %b: tensor<32xf32>) -> tensor<32xf32> {
    %e0 = tensor.empty() : tensor<32xf32>
    %add = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%a, %b : tensor<32xf32>, tensor<32xf32>) outs(%e0 : tensor<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %s = arith.addf %x, %y : f32
        linalg.yield %s : f32
    } -> tensor<32xf32>
    %e1 = tensor.empty() : tensor<32xf32>
    %mul = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%add, %b : tensor<32xf32>, tensor<32xf32>) outs(%e1 : tensor<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %p = arith.mulf %x, %y : f32
        linalg.yield %p : f32
    } -> tensor<32xf32>
    return %mul : tensor<32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // one handle -> two payload generics
    %gens = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.print %gens {name = "both generics (single handle, 2 payloads)"} : !transform.any_op
    // split into two singleton handles, in IR order: #0 = add, #1 = mul
    %first, %second = transform.split_handle %gens
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %first  {name = "split #0 (expect addf generic)"} : !transform.any_op
    transform.print %second {name = "split #1 (expect mulf generic)"} : !transform.any_op
    // climb from a leaf op up to its enclosing func.func
    %parent = transform.get_parent_op %second {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    transform.print %parent {name = "get_parent_op of split #1 (expect func.func @two_gen)"} : !transform.any_op
    transform.yield
  }
}
