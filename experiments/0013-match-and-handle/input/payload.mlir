// Payload only (no transform script).
// @mm: producer(linalg.fill) -> consumer(linalg.matmul) chain.
//      matmul's outs operand (#2) is produced by the fill.
// @two_gen: two linalg.generic ops in one function for split_handle demo.
func.func @mm(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
                     outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

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
