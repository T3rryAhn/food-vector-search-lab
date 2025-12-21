# food_vector_search_lab/query_expansion.py
from __future__ import annotations

import re

"""
짧고 추상적인 쿼리를 의미가 풍부한 문장으로 확장한다.
MVP 단계에서는 규칙 기반 방식 사용.
- '정확히 일치'가 아니라, 문장 내 키워드를 찾아 "부분 확장"한다.
- 영어 hot/warm 같은 토큰은 음료 쪽을 과도하게 끌어올릴 수 있어 최소화한다.
"""

KEYWORD_EXPANSION: list[tuple[re.Pattern, str]] = [
    # 온도
    (re.compile(r"(시원한|차가운)"), "시원한 차가운 아이스"),
    (re.compile(r"(따뜻한|뜨거운|뜨끈한)"), "따뜻한 뜨거운"),
    # 맛/자극
    (re.compile(r"(매콤한|매운)"), "매콤한 매운 자극적인"),
    (re.compile(r"(순한|담백한)"), "순한 담백한 자극적이지 않은"),
    (re.compile(r"(달콤한|달달한)"), "달콤한 단"),
    (re.compile(r"(짭짤한|짜지)"), "짭짤한 짠 나트륨"),
    # 목적/상황
    (re.compile(r"(야식)"), "야식 늦은 밤 간단한 간식 부담 없는"),
    (re.compile(r"(운동\s*후|운동후)"), "운동 후 단백질 회복"),
    (re.compile(r"(혼밥)"), "혼밥 간단한 한그릇"),
    (re.compile(r"(점심)"), "점심 가볍게"),
    # 국물 의도: 카테고리 힌트를 직접 추가(핵심)
    (re.compile(r"(국물|국\b|탕\b|찌개|전골|해장|수프|스프)"), "국 탕 찌개 전골 국물"),
    # 디저트/간식 의도
    (re.compile(r"(디저트|간식)"), "디저트 간식 빵 과자 아이스크림"),
    # 음료 의도
    (re.compile(r"(음료|차\b|커피|라떼|티\b)"), "음료 차 커피"),
]

def expand_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q

    extras: list[str] = []
    for pat, exp in KEYWORD_EXPANSION:
        if pat.search(q):
            extras.append(exp)

    # 중복 토큰 정리(아주 단순하게)
    extra_text = " ".join(extras).strip()
    if not extra_text:
        return q

    return f"{q} {extra_text}"
