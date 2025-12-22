from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

"""
짧고 추상적인 쿼리를 의미가 풍부한 문장으로 확장한다.
MVP 단계에서는 규칙 기반 방식 사용.

정책(A안):
- '정확히 일치'가 아니라 문장 내 키워드를 찾아 "부분 확장"한다.
- kind(food/drink/snack)에 따라 확장 그룹을 제한한다.
  - food 쿼리에서 '커피/라떼/차' 같은 강한 음료 토큰을 추가하면
    HOT/ICED 표면 토큰 때문에 결과가 음료로 쏠리는 부작용이 커서 기본 차단.
- 중복 토큰은 토큰 단위로 제거한다.
"""

# --- helpers --------------------------------------------------------------

_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split((text or "").strip()) if t]


def _dedup_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _is_too_short(q: str) -> bool:
    # 단어 1개(또는 2개라도 너무 짧은 경우)는 과확장 방지
    toks = _tokenize(q)
    if len(toks) <= 1:
        return True
    # "뜨끈한 국물"처럼 2단어라도 짧은 건 과확장 위험이 있어 제한적으로만 확장
    if len(toks) == 2 and len(q) <= 8:
        return True
    return False


# --- expansion rules ------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    expansion: str
    group: str  # temperature / taste / intent / soup / dessert / drink


RULES: list[Rule] = [
    # 온도 (강한 영어 토큰 제거: hot/warm/iced 같은 건 음료 쏠림 유발)
    Rule("temp_cold", re.compile(r"(시원한|차가운|아이스)"), "시원한 차가운", "temperature"),
    Rule("temp_hot", re.compile(r"(따뜻한|뜨거운|뜨끈한)"), "따뜻한 뜨거운", "temperature"),

    # 맛/자극
    Rule("taste_spicy", re.compile(r"(매콤한|매운)"), "매콤한 매운 자극적인", "taste"),
    Rule("taste_mild", re.compile(r"(순한|담백한)"), "순한 담백한", "taste"),
    Rule("taste_sweet", re.compile(r"(달콤한|달달한)"), "달콤한 단", "taste"),
    Rule("taste_salty", re.compile(r"(짭짤한|짜지\s*않|짜지|짠)"), "짭짤한 짠 나트륨", "taste"),

    # 상황/목적
    Rule("intent_night", re.compile(r"(야식)"), "야식 늦은밤 간단한 간식 부담없는", "intent"),
    Rule("intent_postworkout", re.compile(r"(운동\s*후|운동후)"), "운동후 단백질 회복", "intent"),
    Rule("intent_solo", re.compile(r"(혼밥)"), "혼밥 간단한 한그릇", "intent"),
    Rule("intent_lunch", re.compile(r"(점심)"), "점심 가볍게", "intent"),

    # 국물/식사 카테고리 힌트
    Rule("soup_hint", re.compile(r"(국물|국\b|탕\b|찌개|전골|해장|수프|스프)"), "국 탕 찌개 전골 국물", "soup"),

    # 디저트/간식
    Rule("dessert_hint", re.compile(r"(디저트|간식|과자|빵|케이크|아이스크림|빙수)"), "디저트 간식 빵 과자 아이스크림", "dessert"),

    # 음료 (※ 기본적으로 food/snack에서는 확장 비활성화)
    Rule("drink_hint", re.compile(r"(음료|차\b|커피|라떼|티\b|주스|에이드|탄산)"), "음료 차 커피", "drink"),
]

# kind별 허용 그룹
KIND_ALLOWED_GROUPS: dict[str, set[str]] = {
    "food": {"temperature", "taste", "intent", "soup"},       # drink/dessert는 기본 차단
    "snack": {"temperature", "taste", "intent", "dessert"},  # drink/soup 차단
    "drink": {"temperature", "taste", "intent", "drink"},    # soup/dessert 차단
}

# too_short일 때 허용 그룹(과확장 방지)
SHORT_ALLOWED_GROUPS_DEFAULT = {"temperature", "soup", "dessert", "drink"}


def expand_query(query: str, *, kind: Optional[str] = None) -> str:
    """
    query를 규칙 기반으로 확장한다.
    kind를 주면(food/drink/snack), 확장 그룹을 제한한다.

    예)
    - expand_query("뜨끈한 국물", kind="food")
      -> "뜨끈한 국물 따뜻한 뜨거운 국 탕 찌개 전골 국물"
    - expand_query("뜨끈한", kind="food")
      -> "뜨끈한 따뜻한 뜨거운" (너무 짧은 쿼리라 제한적 확장만)
    """
    q = (query or "").strip()
    if not q:
        return q

    # kind 정규화
    k = (kind or "").strip().lower() or None
    allowed_groups = KIND_ALLOWED_GROUPS.get(k, None)

    too_short = _is_too_short(q)
    short_allowed = SHORT_ALLOWED_GROUPS_DEFAULT

    extras: list[str] = []
    for rule in RULES:
        # kind별 그룹 제한
        if allowed_groups is not None and rule.group not in allowed_groups:
            continue
        # 너무 짧은 쿼리일 땐 과확장 방지: 일부 그룹만 허용
        if too_short and rule.group not in short_allowed:
            continue

        if rule.pattern.search(q):
            extras.append(rule.expansion)

    if not extras:
        return q

    # 토큰 단위 중복 제거 (원문 토큰 + 확장 토큰)
    merged = _dedup_tokens(_tokenize(q) + _tokenize(" ".join(extras)))

    return " ".join(merged)
