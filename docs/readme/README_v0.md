# 🍲 Food Semantic Search Experiment

한국어 식단 데이터를 바탕으로, 사용자의 주관적인 속성(예: 든든함, 뜨거움, 무거운 정도)을 수치로 입력받아 최적의 음식을 찾아주는 **의미 기반 추천 시스템** 실험실입니다.

---

## 1. 프로젝트 배경 (Context)
* **데이터 규모**: 한국어 음식 데이터 약 20,000건 (음식명, 카테고리, 영양성분 포함)
    * db 출처 [식품의약품안전처](https://various.foodsafetykorea.go.kr/nutrient/general/down/historyList.do) 
* **핵심 과제**: "뜨거움 0.7 + 무거움 0.5"와 같은 추상적 수치형 쿼리를 어떻게 데이터와 매칭할 것인가?
* **기술적 제약**:
    * **Preprocessing**: Google Colab T4 GPU (무료 요금제)를 활용한 1회성 임베딩
    * **Inference**: 서버 CPU 환경에서 5s 이내 응답 지향
    * **Labeling**: 2만 건의 데이터를 일일이 속성 라벨링할 수 없으므로 모델의 제로샷(Zero-shot) 능력을 활용함

---

## 2. 모델 선정
`2025-12-20` 기준 
허깅페이스 MTEB 리더보드에서 다음 과정을 거쳐 최적의 모델을 선정했습니다.

[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

### 1단계: 필터링 (환경 제약 조건 반영)
먼저 T4 GPU와 한국어 환경에 맞는 모델만 남기기 위해 아래 필터를 적용했습니다.
* Language: Korean (kor) — 한국어 뉘앙스 이해도 필수
* Model Parameters: 1B 이하 — T4 GPU(16GB) 메모리 점유율 및 CPU 추론 속도 최적화 (1B 이상부터는 성능 향상 폭이 적다고 생각했습니다.)

### 2단계: 정렬 (목적 지표 설정)
추천 시스템의 본질인 **사용자의 의도에 맞는 항목 찾기**에 집중하여 지표를 정렬했습니다.
* Primary (내림차순): **Retrieval** — 2만 건 중 질문(속성)에 어울리는 음식을 뽑아내는 검색 성능
* Secondary (참고): **STS** — "든든함"과 "포만감" 등 단어 간 의미 유사도 판별 능력

### 3단계: 선정
> 본 실험에서는 단순한 리더보드 순위가 아니라, 모델의 **범용성(Zero-shot)**과 검색력(Retrieval)을 고려하여 최적의 모델을 선정했습니다.

![MTEB_rank](./docs/static/mteb_rank.png)

#### 1. Snowflake/snowflake-arctic-embed-l-v2.0
- 최고의 검색 효율 (Retrieval 1위): 캡처된 MTEB 한국어 리더보드에서 Retrieval 점수 73.27로 1B 이하 체급 중 가장 정교한 검색 성능을 보유하고 있습니다.


#### 2. jina-embeddings-v3
- (Zero-shot 100%): 리더보드 평가 데이터셋과 학습 데이터셋 사이의 중복이 전혀 없음을 의미하는 zero_shot_percentage 수치가 **100%**입니다.

- 데이터 보완 능력: DB에 '온도(뜨거움/차가움)' 정보가 없는 상황에서, 모델이 보유한 사전 상식(Common Sense)만으로 속성을 유추해야 하는 경우 가장 안정적인 성능을 기대할 수 있습니다.

- 경량화: 메모리 사용량이 1092MB로 snowflake보다 가벼움.

#### 3. 기각 모델: PwC-Embedding_expr
- 낮은 범용성 (Zero-shot 16~25%): zero_shot_percentage 로직(100 - perc_overlap) 분석 결과, 리더보드 평가 데이터의 상당 부분(약 75~84%)을 이미 학습 과정에서 경험한 것으로 파악됩니다.

오버피팅 위험: 학습한 적 없는 새로운 도메인(사용자의 고유 음식 데이터셋)을 만났을 때, 리더보드상의 높은 점수가 재현되지 않을 가능성이 커 실험 후보군에서 제외했습니다.

### 🧪 실험 가설 (Experimental Hypothesis)
- Snowflake 모델은 풍부한 영양소 설명문이 주어질 때 최고의 검색 성능을 보여줄 것이다.
- Jina 모델은 설명이 부족하거나 모호한 쿼리에 대해 모델 특유의 '직관'으로 부족한 속성을 보완하여 추천할 것이다.
- OpenAI Embedding 모델은 유료 API로서 위 로컬 모델들의 성능 지향점을 판단하는 역할을 수행한다.

---

## 3. 핵심 전략: 자동 속성 추출 (Feature Textualization)

모델이 "김치찌개"라는 단어만 보고 주관적 속성을 알 수 없으므로, **영양 성분을 자연어 설명문으로 변환**하여 임베딩하는 전략을 사용합니다.

## 4. 실험 설계

## 5. 참고 문헌 및 링크
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
* [Snowflake Arctic Embed v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)