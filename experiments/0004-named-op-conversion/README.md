# #0004 named-op-conversion — in-tree `linalg-named-op-conversion` 해부·재현

in-tree pass `linalg-named-op-conversion` 의 내부 구성(pass → pattern → 핵심 함수)을
해부하고, 같은 populate 함수 + 같은 driver 를 out-of-tree 에서 호출하는
`my-named-op-conversion` 으로 재현해 byte-diff 로 검증한 실험.

"named → named 정규화" 라는 일반적 이름이지만 LLVM 19.1.7 시점의 실체는
**depthwise conv 2D 한 가족**: channel multiplier 차원(M)이 정적으로 1 인
`depthwise_conv_2d_nhwc_hwcm(_q)` 를 `depthwise_conv_2d_nhwc_hwc(_q)` 로 좁히고
rank 차이를 `tensor.collapse_shape` / `tensor.expand_shape` 로 메운다.

## 호출 체인 (LLVM 19.1.7, 파일:라인)

```
LinalgNamedOpConversionPass                    NamedOpConversions.cpp:145-158
  (def: Passes.td:80-83 — anchor 없는 Pass<"linalg-named-op-conversion">,
   옵션 0개, dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"])
  └─ runOnOperation()                          NamedOpConversions.cpp:151-157
       ├─ populateLinalgNamedOpConversionPatterns(patterns)
       │       선언 Transforms.h:1711 / 정의 NamedOpConversions.cpp:161-165
       │    └─ patterns.add<SimplifyDepthwiseConvOp,
       │                    SimplifyDepthwiseConvQOp>(ctx)       (pattern 2개)
       │         SimplifyDepthwiseConvOp  : OpRewritePattern<DepthwiseConv2DNhwcHwcmOp>
       │                                                          :104-122
       │         SimplifyDepthwiseConvQOp : OpRewritePattern<DepthwiseConv2DNhwcHwcmQOp>
       │                                                          :124-143
       │         └─ (둘 다) matchAndReplaceDepthwiseConv(...)     :35-101
       │              static LogicalResult matchAndReplaceDepthwiseConv(
       │                Operation *, Value input, Value kernel, Value iZp,
       │                Value kZp, Value init, Attribute stride,
       │                Attribute dilation, PatternRewriter &)
       │              (비양자화 pattern 은 iZp/kZp = nullptr 로 호출 :118-120,
       │               양자화 pattern 은 DPS input 2,3 을 전달     :140-141)
       └─ applyPatternsAndFoldGreedily(op, std::move(patterns))   :155
```

## matchAndReplaceDepthwiseConv 내부 단계 ↔ IR 변화 매핑

| 코드 단계 (NamedOpConversions.cpp) | IR 에 만드는 차이 |
|---|---|
| `:40-43` `hasPureTensorSemantics()` 아니면 bail | memref 버전 conv 는 불변 (negative (3)) — 변환이 만드는 collapse/expand 가 tensor op 이기 때문 |
| `:47-51` kernel/init/result 가 `RankedTensorType` 아니면 bail | unranked 입력 불변 |
| `:53-54` `kernelTy.getDimSize(3) != 1` 이면 bail | M=2 (negative (1)), M=`?` dynamic (negative (2)) 모두 불변 — 판정은 **정적** shape 기준 (`kDynamic != 1`) |
| `:56-63` kernel `tensor.collapse_shape [[0],[1],[2,3]]` 생성 | `tensor<3x3x96x1xf32> into tensor<3x3x96xf32>` — HWCM→HWC, M 차원을 C 에 흡수 |
| `:65-74` init `tensor.collapse_shape [[0],[1],[2],[3,4]]` 생성 | `tensor<1x56x56x96x1xf32> into tensor<1x56x56x96xf32>` — NHWCM→NHWC |
| `:76-91` `TypeSwitch` 로 새 named op 생성 (`DepthwiseConv2DNhwcHwcmOp`→`DepthwiseConv2DNhwcHwcOp`, `..HwcmQOp`→`..HwcQOp`; 그 외 nullptr→failure) | op 이름이 `linalg.depthwise_conv_2d_nhwc_hwc(_q)` 로 바뀌고 operand 가 collapse 결과로 대체. iZp/kZp 스칼라는 그대로 통과 |
| `:80,86` `getPrunedAttributeList(op)` (Utils.h:368-374) → `:94-95` 새 op 에 재부착 | discardable attr (`_someattr`) 보존; 정의된 attr (strides/dilations) 는 builder 가 새로 설정, memoized indexing maps 는 제외 |
| `:97-99` `replaceOpWithNewOp<tensor::ExpandShapeOp>` (같은 reassociation) | 4-D conv 결과를 원래 5-D 로 복원. **dynamic shape 이면** ExpandShapeOp 의 reassociation-only builder (TensorOps.cpp:1687-1701) 가 `inferOutputShape` (:1663-1674) 로 `arith.constant index` + `tensor.dim` 4개를 추가 생성해 `output_shape [%dim, %dim_1, %dim_2, %dim_3, 1]` 을 채운다 — NamedOpConversions.cpp 에는 없는 op 들이 builder 에서 나오는 표본 |

조건 요약 — 발화하려면 **(a) pure tensor semantics, (b) kernel/init/result 가
ranked tensor, (c) kernel 의 dim 3 (multiplier M) 이 정적으로 1** 이어야 한다.
대상 op 은 정확히 2종: `depthwise_conv_2d_nhwc_hwcm` → `..._hwc`,
`depthwise_conv_2d_nhwc_hwcm_q` → `..._hwc_q`.

## 입력/결과

| 입력 | 기대 | 실제 (output.* = intree.* byte-identical) |
|---|---|---|
| `input/depthwise-m1.mlir` | M=1 정적/dynamic-나머지 2케이스 발화 | `hwcm`→`hwc` + collapse×2 + expand; dynamic 함수에는 `tensor.dim`×4 + `arith.constant`×4 추가 (ExpandShapeOp builder 산물) |
| `input/depthwise-m1-q.mlir` | 양자화 변형 발화 | `hwcm_q`→`hwc_q`, iZp/kZp (i32) 는 ins 에 그대로 통과 |
| `input/negative.mlir` | 불변 | (1) M=2, (2) M=`?`, (3) memref semantics — 모두 op 구조 불변 |

## 재현

```bash
./run.sh
# [OK ] byte-identical : depthwise-m1
# [OK ] byte-identical : depthwise-m1-q
# [OK ] byte-identical : negative
```

out-of-tree 재현 pass: `out-of-tree/lib/Passes/MyNamedOpConversion.cpp`
(`my-named-op-conversion`). in-tree `runOnOperation()` 과 동일하게
`linalg::populateLinalgNamedOpConversionPatterns` + `applyPatternsAndFoldGreedily`
호출만 한다 (알고리즘 재구현 없음). link lib 는 기존
`MLIRLinalgTransforms`(populate/pattern 정의) + `MLIRTransforms`(greedy driver) 에
`MLIRTensorDialect` 를 명시 추가 (dependent dialect 로 `tensor::TensorDialect` 직접 참조).

## 이식 메모 (개인 컴파일러 반영 시)

- 가져갈 것: static 함수 `matchAndReplaceDepthwiseConv` (~65줄) + 얇은 pattern
  2개. 의존: `LinalgOp::hasPureTensorSemantics`, named op 4종
  (`DepthwiseConv2DNhwcHwcm(Q)Op` / `DepthwiseConv2DNhwcHwc(Q)Op`) 의 builder,
  `tensor::CollapseShapeOp`/`ExpandShapeOp` (reassociation builder),
  `getPrunedAttributeList` (Utils.h template).
- pass 골격은 보일러플레이트: populate 한 줄 + greedy driver 한 줄.
- dependentDialects 에 **tensor 를 반드시 선언** — pattern 이 tensor op 를 새로
  만든다 (#0003 과 달리 linalg 의 의존만으로 안전하다고 가정하지 않고 in-tree
  def 그대로 따름). expand 측에서 dynamic shape 면 arith.constant 도 생기는데
  arith 는 tensor dialect 의 의존으로 따라 로드된다.
- multiplier 판정이 정적 기준임에 주의 — shape inference 가 앞서 돌아 M 차원을
  1 로 굳혀줘야 발화 범위가 넓어진다.
