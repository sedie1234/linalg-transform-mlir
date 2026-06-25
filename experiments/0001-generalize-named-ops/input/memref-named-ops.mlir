// #0001 입력 2 (positive, buffer/memref semantics)
// hasPureTensorSemantics == false 이므로 generalizeNamedOp 의
// resultTypes = TypeRange{} 분기 (Generalization.cpp:62-64) 를 탄다 —
// 생성되는 linalg.generic 은 결과값이 없는 (메모리 부수효과만 있는) 형태.
func.func @memref_named(%A: memref<4x8xf32>, %B: memref<8x16xf32>,
                        %C: memref<4x16xf32>, %cst: f32) {
  // fill: scalar input 하나, output 으로 broadcast 쓰기
  linalg.fill ins(%cst : f32) outs(%C : memref<4x16xf32>)
  // matmul on memref: 결과 타입 없는 generic 으로
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x16xf32>)
                outs(%C : memref<4x16xf32>)
  return
}
