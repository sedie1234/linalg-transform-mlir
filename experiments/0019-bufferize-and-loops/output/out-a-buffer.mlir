module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %alloc = memref.alloc() : memref<128x64xf32>
    bufferization.materialize_in_destination %arg2 in writable %alloc : (tensor<128x64xf32>, memref<128x64xf32>) -> ()
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x64xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    memref.dealloc %alloc : memref<128x64xf32>
    return %1 : tensor<128x64xf32>
  }
}

