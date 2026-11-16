# 실서비스 음식 검색 개선 기록

기존 임베딩 연구를 실제 메뉴 선택 경험으로 이어가는 독립 실험 트랙이다. 기존 `src/` CLI의 실행 결과와 운영 서비스 결과를 같은 구현의 결과로 취급하지 않는다.

## 코드와 기록의 경계

| 영역 | 기존 Lab | 실서비스 트랙 |
|---|---|---|
| 목적 | 여러 임베딩 모델·영양 재랭킹 연구 | 실제 메뉴 선택 과정의 검색 품질 개선 |
| 실행 | src/cli.py | 별도 EATIT AI 서비스의 FastAPI |
| 인덱스 메타 | 모델별 .faiss + .parquet | food_faiss.index + food_meta.pkl |
| 모델 | Snowflake/Jina/OpenAI 선택 가능 | 현재 Snowflake 로컬 CPU |
| 의존성 예 | sentence-transformers5.2.0, torch2.9.1 | 배포 시4.1.0, CPU torch2.6.0 |
| 현재 보관 | 기존 코드·과거 보고서 유지 | 입력·관측 결과·출처·판단 기록 |

운영 코드의 출발점은 [EATIT_back_ai](https://github.com/DevOops-Korea/EATIT_back_ai)다. 별도 배포 overlay가 존재하므로 그 저장소 commit만 checkout해도 이번 결과가 재현된다고 주장하지 않는다. 이 Lab에 운영 앱/배포 설정/인덱스를 복제하지 않는다. 다음 구현 실험 전에 서비스 snapshot 또는 명시적 adapter를 선택하고, 별도 환경에서 기준선을 먼저 재현한다. 기존 src 모듈을 묵시적으로 import하지 않는다.

## 읽는 순서

1. [운영 중 발견한 문제와 기준선](runs/2026-11-16-baseline/report.md)
2. [실험 원본 결과](runs/2026-11-16-baseline/results.json), [코드·모델 출처와 한계](runs/2026-11-16-baseline/manifest.json)
3. [저장소 경계에 대한 결정](decisions/001-separate-service-track.md)
4. [다음 실험 계획](plan.md)

## 기록 원칙

- 발견 → 가설 → 통제한 비교 → 관측 → 채택/기각 → 운영 적용 여부 순서로 기록한다.
- 계획은 실험 결과가 아니다. 기준선과 첫 개선 비교/배포를 완료했다. 독립 평가셋 기반 정확도는 아직 없다.
- 검색어는 작성한 가상 입력이다. 방문자 검색 기록, IP, 개인별 선택, 인증/운영 접근 정보는 넣지 않는다.
- 원본 run은 고정한다. 재실행은 새 run 디렉터리에 기록한다.
- 코드/모델/데이터/인덱스 버전, query prompt, 후보 수, 필터/다양화 설정, 평가셋, 지연 측정 범위를 함께 기록한다. 확인하지 못한 것은 미확인으로 남긴다.
- 주관 평가는 누가 어떤 기준으로 했는지 명시한다. 현재 평가는 개발 보조 에이전트의 판단이며 사용자 설문이나 정답셋 기반 정확도가 아니다.
- 서비스 평가와 Lab 모델 비교는 서로 다른 구현이다. 구현 차이를 통제하기 전에는 직접 우열을 비교하지 않는다.

새 run에는 `manifest.json`, `results.json`, `report.md`를 만든다. 이번 `cases.json`은 탐색용 기준선으로 유지하고, 이를 보고 수정한 후에는 별도 미사용 표현/예외 셋으로 과적합도 확인한다.

## 첫 개선 결과와 실행

[비교·기각·채택 기록](runs/2026-11-16-improvement/report.md)을 추가했다. 실행 도구는 이 트랙의 tools 아래에만 있으며 기존 src CLI와 분리했다.

```sh
# 버전이 맞는 서비스 의존성/모델 캐시 환경에서, 운영 프로세스와 별도로 실행
python experiments/service-retrieval/tools/compare_service.py --service-root /path/to/service-checkout --cases experiments/service-retrieval/comparison-cases.json --enhanced > new-run.json
```

모델/데이터/인덱스를 재생성하지 않는다. API 집계도 호출하지 않는다. 실행 시 service source의 query_expansion/diversify가 기준선 manifest와 맞는지 확인해야 한다. 서비스 모듈을 import하므로 실제 운영 파일 대신 고정 checkout 사용을 권장한다. 결과의 hashes와 도구 commit을 함께 보관한다. 모델이 메모리에 한 번 더 로드되므로 자원 여유가 있는 환경에서 순차 실행한다.
