from __future__ import annotations
import re
import pandas as pd
from .category_guardrails import QueryKind, kind_of_cat_large

CHESTNUT_PATTERNS = [
    r"통밤", r"밤식빵", r"뜯어먹는밤식빵", r"밤밥", r"밤고구마", r"밤호박",
    r"밤\s*빵", r"밤\s*케이크", r"밤\s*타르트",
]

def apply_category_guardrails(
    hits: pd.DataFrame,
    *,
    kind: QueryKind,
    query_text: str,
    penalty_other: float = 0.25,
    bonus_match: float = 0.05,
    penalty_chestnut: float = 0.35,
) -> pd.DataFrame:
    """
    cat_large(=CSV 카테고리) 기반으로 점수 보정.
    - kind는 프론트에서 택1로 받아야 함: food/drink/snack
    - query_text는 야식 밤(栗) 트랩 같은 추가 규칙에만 사용
    """
    if hits is None or len(hits) == 0:
        return hits
    df = hits.copy()

    if "score" not in df.columns:
        return df

    cat_large = df["cat_large"].fillna("").astype(str)
    name = df["food_name"].fillna("").astype(str)

    # 1) kind 기반: 매칭 카테고리 소폭 가점, 그 외 감점
    is_match = cat_large.apply(lambda x: kind_of_cat_large(x) == kind)
    df.loc[is_match, "score"] = df.loc[is_match, "score"] + bonus_match
    df.loc[~is_match, "score"] = df.loc[~is_match, "score"] - penalty_other

    # 2) 야식 밤(栗) 트랩: query에 '야식/밤에/심야/출출'이 있으면 밤(栗) 관련을 강감점
    q = (query_text or "").strip()
    if any(k in q for k in ["야식", "밤에", "심야", "출출", "늦은 밤", "야밤"]):
        pat = "(" + "|".join(CHESTNUT_PATTERNS) + ")"
        is_chestnut = name.str.contains(pat, regex=True)
        df.loc[is_chestnut, "score"] = df.loc[is_chestnut, "score"] - penalty_chestnut

    return df.sort_values("score", ascending=False).reset_index(drop=True)
