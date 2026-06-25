#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_buf(%arg0: memref<128x256xf32>, %arg1: memref<256x64xf32>, %arg2: memref<128x64xf32>) {
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
          %0 = affine.apply #map(%arg3, %arg4, %arg5)
          %1 = affine.apply #map1(%arg3, %arg4, %arg5)
          %2 = memref.load %arg0[%0, %1] : memref<128x256xf32>
          %3 = affine.apply #map1(%arg3, %arg4, %arg5)
          %4 = affine.apply #map2(%arg3, %arg4, %arg5)
          %5 = memref.load %arg1[%3, %4] : memref<256x64xf32>
          %6 = affine.apply #map(%arg3, %arg4, %arg5)
          %7 = affine.apply #map2(%arg3, %arg4, %arg5)
          %8 = memref.load %arg2[%6, %7] : memref<128x64xf32>
          %9 = arith.mulf %2, %5 : f32
          %10 = arith.addf %8, %9 : f32
          %11 = affine.apply #map(%arg3, %arg4, %arg5)
          %12 = affine.apply #map2(%arg3, %arg4, %arg5)
          memref.store %10, %arg2[%11, %12] : memref<128x64xf32>
        }
      }
    }
    return
  }
}

