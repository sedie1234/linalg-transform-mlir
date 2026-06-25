module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    affine.for %arg3 = 0 to %dim {
      affine.for %arg4 = 0 to %dim_1 {
        affine.for %arg5 = 0 to %dim_0 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<?x?xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
        }
      }
    }
    return
  }
}

