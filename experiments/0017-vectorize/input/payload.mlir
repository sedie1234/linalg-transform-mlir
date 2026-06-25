// T07 payload — static-shape ops to vectorize.
// vectorize 는 보통 static shape 필요. 두 종류 제공:
//   @matmul    : static matmul (linalg.matmul -> vector.contract / outerproduct)
//   @add_elemwise : static elementwise add (linalg.generic -> vector.transfer + arith)
func.func @matmul(%A: tensor<8x16xf32>, %B: tensor<16x4xf32>,
                  %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<8x16xf32>, tensor<16x4xf32>)
                     outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

#map = affine_map<(d0, d1) -> (d0, d1)>
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
