#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = bufferization.to_memref %arg1 : memref<256x64xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<128x256xf32, strided<[?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<128x64xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<128x64xf32>
    memref.copy %2, %alloc : memref<128x64xf32, strided<[?, ?], offset: ?>> to memref<128x64xf32>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c128 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c64 step %c1_1 {
        %c0_2 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg5 = %c0_2 to %c256 step %c1_3 {
          %4 = affine.apply #map(%arg3, %arg4, %arg5)
          %5 = affine.apply #map1(%arg3, %arg4, %arg5)
          %6 = memref.load %1[%4, %5] : memref<128x256xf32, strided<[?, ?], offset: ?>>
          %7 = affine.apply #map1(%arg3, %arg4, %arg5)
          %8 = affine.apply #map2(%arg3, %arg4, %arg5)
          %9 = memref.load %0[%7, %8] : memref<256x64xf32, strided<[?, ?], offset: ?>>
          %10 = affine.apply #map(%arg3, %arg4, %arg5)
          %11 = affine.apply #map2(%arg3, %arg4, %arg5)
          %12 = memref.load %alloc[%10, %11] : memref<128x64xf32>
          %13 = arith.mulf %6, %9 : f32
          %14 = arith.addf %12, %13 : f32
          %15 = affine.apply #map(%arg3, %arg4, %arg5)
          %16 = affine.apply #map2(%arg3, %arg4, %arg5)
          memref.store %14, %alloc[%15, %16] : memref<128x64xf32>
        }
      }
    }
    %3 = bufferization.to_tensor %alloc : memref<128x64xf32>
    memref.dealloc %alloc : memref<128x64xf32>
    return %3 : tensor<128x64xf32>
  }
}

