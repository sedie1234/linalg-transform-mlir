# linalg + transform MLIR — 학습 기록

MLIR의 **linalg dialect 패스 내부구조**와 **transform dialect("외부 schedule")**를 직접 실행·관찰하며 정리한 학습 저장소. 모든 IR·측정은 **LLVM/MLIR 19.1.7**(`llvmorg-19.1.7`) 기준 실측이다.

각 실험은 `payload.mlir`(변환 대상) + `schedule`/`pass` + `output/`(변환 후 IR) + `run.sh`(재현 명령)로 구성되며, "같은 입력에 변환만 바꿔 IR이 어떻게 달라지는가"를 눈으로 확인할 수 있게 했다.

## 무엇을 알 수 있나

### P1 — in-tree linalg 패스 해부 (work 0001~0010)
대표적인 in-tree linalg 패스 10종을 (a) **호출 체인**(pass → pattern → 핵심 함수, 파일:라인) 해부, (b) **out-of-tree 재현**(같은 함수를 가져와 내 패스로 호출 → in-tree와 byte-diff), (c) **IR 전/후** 주석으로 정리:

`generalize-named-ops` · `specialize-generic-ops` · `inline-scalar-operands` · `named-op-conversion` · `convert-elementwise-to-linalg` · `fuse-elementwise-ops` · `fold-unit-extent-dims` · `detensorize` · `convert-linalg-to-{loops,affine,parallel}` · `block-pack-matmul`

→ "linalg 패스가 내부에서 실제로 무슨 함수를 부르고, IR을 어떻게 바꾸며, 그걸 out-of-tree에서 어떻게 재현하는가"를 알 수 있다.

### P2 — transform dialect "외부 schedule" (work 0011~0022 / T01~T12)
변환(=schedule)을 컴파일러 *밖*의 IR로 명시하고 `-transform-interpreter`로 적용하는 방식. 같은 payload에 schedule만 바꿔 결과 IR을 비교:

| work | 주제 | 핵심 |
|------|------|------|
| 0011 | 멘탈 모델 + 첫 schedule | payload/transform IR/handle/interpreter, `match`+`tile_using_for` |
| 0012 | 고정 패스 vs transform script | 같은 `generalize`를 두 방식으로 |
| 0013 | match & handle | `split_handle`·`get_producer_of_operand`·`get_parent_op` |
| 0014 | tile (for vs forall) | `scf.for`(순차) vs `scf.forall`(병렬) |
| 0015 | fuse | producer를 tile 루프로, 중간 텐서 materialize 회피 |
| 0016 | pad & pack | blocked 4D layout(mmt4d), `pack`/`pack_transpose`/`lower_pack` |
| 0017 | vectorize | linalg → vector dialect (`multi_reduction`/`contract`) |
| 0018 | 표현 변환 | `generalize`·`specialize`·`interchange`·`decompose` |
| 0019 | bufferize & to-loops | tensor → memref(in-place vs RaW copy) → `scf.for` |
| 0020 | full kernel schedule | fc+bias+relu를 match→tile→fuse→vectorize 풀 파이프라인 |
| 0021 | schedule sweep | tile size만 바꿔 다른 IR — 재빌드 없는 탐색(autotuning 스케치) |
| 0022 | 제어흐름·매처 | `foreach`·`alternatives`·`match.operation_name`, 명령형 vs greedy(`apply_patterns`) |

→ "변환 규칙을 컴파일러 밖 IR로 분리해 재빌드 없이 적용·교체한다는 것이 무엇이고 어떤 이점이 있는가"를 실측으로 알 수 있다.

## 구조

| 경로 | 용도 |
|------|------|
| `experiments/NNNN-<slug>/` | work별 `input/`(payload·schedule) · `output/`(변환 후 IR) · `run.sh`(재현) |
| `docs/` | 학습 노트(HTML). `docs/linalg-pass-internals.html` = P1, `docs/transform-dialect/` = P2 |
| `log/` | work별 5섹션 기록(목적·수행·예상·실제·결론) |
| `out-of-tree/` | in-tree 패스를 재현하는 out-of-tree MLIR 패스 + `my-mlir-opt` (소스만; `build/` 제외) |

## 사용법

전제: **LLVM/MLIR 19.1.7** 빌드(`mlir-opt`)가 있어야 한다(예제 IR이 이 버전 기준).

```bash
# 1) transform dialect 예제 재현 — 같은 payload에 schedule만 바꿔 IR 비교
cd experiments/0011-transform-tile-first
MLIR_OPT=/path/to/llvm-19.1.7/bin/mlir-opt ./run.sh
#   input/  = payload + schedule(.mlir),  output/ = 변환 후 IR

# 2) P1 패스 + out-of-tree 재현 빌드 (LLVM 19.1.7 빌드 트리 필요)
cmake -S out-of-tree -B out-of-tree/build -DMLIR_DIR=<llvm>/lib/cmake/mlir
cmake --build out-of-tree/build           # → out-of-tree/build/bin/my-mlir-opt
```

- 각 `experiments/NNNN/run.sh`는 사용한 `mlir-opt` 명령과 입력→출력 매핑을 그대로 담는다 — 그 파일만 보면 무엇을 어떻게 돌렸는지 재현 가능하다.
- 정리된 설명은 `docs/`의 HTML 노트를 브라우저로 열어 본다(코드 하이라이팅 + 변환 전/후 강조 포함).
