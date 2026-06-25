# Linalg Transform MLIR Work 로그 인덱스

이 디렉토리는 워크스페이스에서 진행된 모든 **work의 영구 기록**이다.
모든 로그는 `linalg-transform-mlir-log-keeper` agent + `work-log` skill을 통해 작성된다.

> (2026-06-12 재시작) 현 cycle = **in-tree linalg pass 내부구조 학습** (`docs/pass-internals-plan.html`).
> 이전 cycle(transform-API spine, 구 0001~0029) 로그는 `../archive/2026-06-cycle1-transform-api/log/`에 보관 — 번호는 0001부터 재시작.

## 트리

```
log/
├── README.md          ← 이 파일 (인덱스)
├── assets/
│   └── work-log.css   ← 모든 HTML 로그가 link 하는 공통 스타일
└── NNNN-<slug>.html   ← 개별 work 로그 (기본 HTML, 필요 시 .md)
```

- **번호 할당 규칙**: 4자리 zero-padding, 단일 글로벌 시퀀스 (재시작으로 0001부터).
- **slug**: 영문 소문자 + 하이픈, 짧고 구체적으로.
- **연관 코드**: 각 로그는 동일 번호의 `experiments/NNNN-<slug>/` 폴더와 1:1 매칭.

## 로그 포맷 (고정 5섹션 — work-log skill이 강제)

```markdown
---
id: NNNN
title: <한 줄 제목>
status: done | failed | partial
date: YYYY-MM-DD
tags: [...]
related_code: experiments/NNNN-<slug>/   # 코드 없으면 생략
---

# NNNN. <제목>

## 1. 무엇을 하고자 하는지
## 2. 수행한 일
## 3. 예상되는 결과
## 4. 실제 결과
## 5. 결론
```

## 인덱스 (시간 역순)

| # | 제목 | 상태 | 날짜 | 태그 |
|---|------|------|------|------|
| 0010+ | [linalg-block-pack-matmul 옵션별 before/after (인터랙티브)](0010-block-pack-matmul-options/index.html) | done | 2026-06-18 | linalg, block-pack-matmul, 옵션별, before-after-diff, tensor-pack, mmt4d, transpose-blocks, allow-padding, mnk-order, mnk-padded-multiples, 실제-pass-출력, 자체완결-html, 비호스팅 |
| 0007+ | [linalg-fold-unit-extent-dims 심화 노트 (인터랙티브)](0007-fold-unit-extent-dims-deepdive/index.html) | done | 2026-06-18 | linalg, fold-unit-extent-dims, 심화, 인터랙티브, 애니메이션, before-after-diff, 함수·옵션·4케이스·이점, 자체완결-html, 비호스팅 |
| 0010 | [linalg-block-pack-matmul 해부·재현·관찰](0010-block-pack-matmul.html) | done | 2026-06-12 | linalg, pass-internals, block-pack-matmul, tensor-pack, mmt4d, data-layout, control-fn, pack-transpose, greedy-driver, byte-diff |
| 0009 | [convert-linalg-to-loops 3종 해부·재현·관찰](0009-linalg-to-loops-trio.html) | done | 2026-06-12 | linalg, pass-internals, linalg-to-loops, scf-for, affine-for, scf-parallel, generate-loop-nest, indexing-maps, greedy-driver, byte-diff |
| 0008 | [linalg-detensorize 해부·재현·관찰](0008-detensorize.html) | done | 2026-06-12 | linalg, pass-internals, detensorize, interface-pass, dialect-conversion, full-conversion, type-converter, materialization, cost-model, byte-diff |
| 0007 | [linalg-fold-unit-extent-dims 해부·재현·관찰](0007-fold-unit-extent-dims.html) | done | 2026-06-12 | linalg, pass-internals, unit-extent-dims, rank-reduction-strategy, indexing-map-역사상, collapse-expand-shape, extract-insert-slice, greedy-driver, byte-diff |
| 0006 | [linalg-fuse-elementwise-ops 해부·재현](0006-fuse-elementwise-ops.html) | done | 2026-06-12 | linalg, pass-internals, elementwise-fusion, indexing-map-합성, control-fusion-fn, reshape-propagation, greedy-driver, byte-diff |
| 0005 | [convert-elementwise-to-linalg 해부·재현·관찰](0005-elementwise-to-linalg.html) | done | 2026-06-12 | linalg, pass-internals, elementwise-to-linalg, elementwise-mappable-trait, dialect-conversion, conversion-target, match-any-op-pattern, byte-diff |
| 0004 | [linalg-named-op-conversion 해부·재현·관찰](0004-named-op-conversion.html) | done | 2026-06-12 | linalg, pass-internals, named-op-conversion, depthwise-conv, collapse-expand-shape, pattern-pass, greedy-driver, byte-diff |
| 0003 | [linalg-inline-scalar-operands 해부·재현·관찰](0003-inline-scalar-operands.html) | done | 2026-06-12 | linalg, pass-internals, inline-scalar-operands, affine-map-isconstant, tensor-extract, pattern-pass, greedy-driver, byte-diff |
| 0002 | [linalg-specialize-generic-ops 해부·재현·관찰](0002-specialize-generic-ops.html) | done | 2026-06-12 | linalg, pass-internals, specialization, idiom-recognition, pattern-pass, greedy-driver, byte-diff |
| 0001 | [linalg-generalize-named-ops 해부·재현·관찰](0001-generalize-named-ops.html) | done | 2026-06-12 | linalg, pass-internals, generalization, pattern-pass, greedy-driver, byte-diff |
