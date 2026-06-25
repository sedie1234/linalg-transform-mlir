module {
  func.func @rowsum_with_index(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 8 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<4x8xf32>
        %1 = affine.load %arg1[%arg2] : memref<4xf32>
        %2 = arith.addi %arg2, %arg3 : index
        %3 = arith.index_cast %2 : index to i64
        %4 = arith.sitofp %3 : i64 to f32
        %5 = arith.addf %0, %4 : f32
        %6 = arith.addf %1, %5 : f32
        affine.store %6, %arg1[%arg2] : memref<4xf32>
      }
    }
    return
  }
}

