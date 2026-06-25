# Experiments

각 work에서 작성/실행한 실제 코드.

## 규칙

- 폴더 이름: `NNNN-<slug>/` — 동일 번호의 로그(`log/NNNN-<slug>.md`)와 1:1 매칭.
- 폴더 안 구조는 자유. 보통:
  ```
  NNNN-<slug>/
  ├── README.md       # 무엇을 하는 코드인지 한 줄 (선택)
  ├── main.<ext>      # 메인 코드
  ├── ref.<ext>       # 비교용 (있을 때)
  └── run.sh          # 빌드/실행 명령 (선택)
  ```
- 빌드 산출물은 `.gitignore` 또는 커밋 안 함.
