module {
  func.func @rowsum_with_index(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<4x8xf32>
        %1 = memref.load %arg1[%arg2] : memref<4xf32>
        %2 = arith.addi %arg2, %arg3 : index
        %3 = arith.index_cast %2 : index to i64
        %4 = arith.sitofp %3 : i64 to f32
        %5 = arith.addf %0, %4 : f32
        %6 = arith.addf %1, %5 : f32
        memref.store %6, %arg1[%arg2] : memref<4xf32>
      }
      scf.reduce 
    }
    return
  }
}

