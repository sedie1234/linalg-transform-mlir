// Payload only (no transform script).
// Two fusion-worthy producer->consumer chains:
//
//  @mm_chain    : elemwise_binary add (producer) -> matmul (consumer/root)
//  @elem_chain  : elemwise_unary (producer) -> elemwise_binary (consumer/root)
//
// In both, the producer's result feeds an `ins` of the consumer, so tiling the
// consumer and fusing pulls the producer slice into the consumer's loop nest.

func.func @mm_chain(%A: tensor<128x256xf32>, %A2: tensor<128x256xf32>,
                    %B: tensor<256x64xf32>, %C: tensor<128x64xf32>)
    -> tensor<128x64xf32> {
  // producer: elementwise add  AA = A + A2
  %AA = linalg.elemwise_binary
      ins(%A, %A2 : tensor<128x256xf32>, tensor<128x256xf32>)
      outs(%A : tensor<128x256xf32>) -> tensor<128x256xf32>
  // consumer / root: matmul AA * B
  %0 = linalg.matmul
      ins(%AA, %B : tensor<128x256xf32>, tensor<256x64xf32>)
      outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}

func.func @elem_chain(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // producer: elementwise unary
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<512x512xf32>)
                             outs(%arg1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  // consumer / root: elementwise binary consuming the producer result
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<512x512xf32>, tensor<512x512xf32>)
                              outs(%arg1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %1 : tensor<512x512xf32>
}
