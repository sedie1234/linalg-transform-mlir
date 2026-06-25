# Docs

학습하면서 정리하는 **개념 노트**.

로그(`log/`)와의 차이:
- **로그**: "이 work에서 무엇을 했고 어떤 결과가 나왔는가" — 시간순/실험 단위.
- **docs**: "이 개념은 무엇이고 어떻게 동작하는가" — 주제별/정제된 형태.

여러 로그에서 얻은 인사이트가 쌓이면 docs로 승격한다 (또는 Plane page로).

## 현재 docs

- **[linalg-pass-internals.html](linalg-pass-internals.html)** — **pass별 해부 카탈로그** (cycle #0001–#0010 누적 핵심 산출물, 2026-06-12): in-tree linalg pass 12종 전수의 호출 체인(파일:라인)·driver 종류·옵션 흐름·핵심 함수 시그니처·IR 전후 발췌·**이식 가이드(#include/호출 절차/link lib)**. byte-diff 합산 69/69 identical.
- **[p1-execution-report.html](p1-execution-report.html)** — P1 실행 보고 (2026-06-12): 10 cycle done, byte-diff 69/69 identical + 검증자 독립 재현 10/10, 구조적 발견(greedy 8 / dialect-conversion 2).
- **[pass-internals-plan.html](pass-internals-plan.html)** — 현 학습 계획 (2026-06-12 재시작): in-tree linalg pass 12종의 내부 코드 구성 + IR 전후 변화 해부, out-of-tree 재현으로 개인 컴파일러 이식 검증. 10 cycle.
- `learning-motivation-from-affine-mlir.md` — affine→linalg 학습 동기 노트.

## 보관

- 이전 cycle(transform-API spine 학습) 산출물: `../archive/2026-06-cycle1-transform-api/`
