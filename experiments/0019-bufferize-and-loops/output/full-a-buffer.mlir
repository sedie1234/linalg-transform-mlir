module attributes {transform.with_named_sequence} {
  func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %alloc = memref.alloc() : memref<128x64xf32>
    bufferization.materialize_in_destination %arg2 in writable %alloc : (tensor<128x64xf32>, memref<128x64xf32>) -> ()
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x64xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    memref.dealloc %alloc : memref<128x64xf32>
    return %1 : tensor<128x64xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %allocated_buffer, %new_ops = transform.structured.bufferize_to_allocation %0 {bufferize_destination_only, emit_dealloc, memory_space = 0 : i64} : !transform.any_op
    transform.yield 
  }
}

