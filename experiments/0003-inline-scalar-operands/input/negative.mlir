// negative 케이스 — pass 가 발화하지 않아 IR 불변이어야 한다.
//
// (1) memref(buffer) semantics: indexing map 은 상수( () )지만
//     hasPureTensorSemantics() 가 false 이므로 matchAndRewrite :38-39 에서 bail.
#map_scalar = affine_map<(d0) -> ()>
#map_id     = affine_map<(d0) -> (d0)>

func.func @no_inline_memref(%scalar: memref<f32>, %x: memref<8xf32>, %out: memref<8xf32>) {
  linalg.generic
      {indexing_maps = [#map_scalar, #map_id, #map_id],
       iterator_types = ["parallel"]}
      ins(%scalar, %x : memref<f32>, memref<8xf32>)
      outs(%out : memref<8xf32>) {
  ^bb0(%s: f32, %xi: f32, %o: f32):
    %sum = arith.addf %s, %xi : f32
    linalg.yield %sum : f32
  }
  return
}

// (2) tensor semantics 이지만 모든 input 의 map 이 비상수(d0 포함):
//     scalarOperands 가 비어 matchAndRewrite :54-55 에서 bail.
func.func @no_inline_nonconst(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %res = linalg.generic
      {indexing_maps = [#map_id, #map_id, #map_id],
       iterator_types = ["parallel"]}
      ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%ai: f32, %bi: f32, %o: f32):
    %m = arith.mulf %ai, %bi : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %res : tensor<8xf32>
}
