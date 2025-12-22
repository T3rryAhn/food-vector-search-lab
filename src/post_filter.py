from __future__ import annotations

import re
import pandas as pd


# kind -> 허용 cat_large 후보(가드레일용)
# ※ food_data.csv의 cat_large 값을 기준으로 작성
KIND_ALLOWED_CAT_LARGE = {
    "food": [
        # 식사/반찬/조리식 중심 (예시)
        "밥류",
        "면 및 만두류",
        "국 및 탕류",
        "찌개 및 전골류",
        "구이류",
        "볶음류",
        "찜 및 조림류",
        "전 및 부침류",
        "나물 및 생채류",
        "김치류",
        "샐러드류",
        "소스/드레싱",
        "기타 음식",
        "곡류 및 서류",
        "어패류",
        "육류",
        "채소류",
        "두류",
        "알류",
    ],
    "drink": [
        "음료 및 차류",
        "커피",
        "차류",
        "주스",
        "탄산음료",
        "기타 음료",
    ],
    "snack": [
        "빵 및 과자류",
        "디저트",
        "아이스크림",
        "떡류",
        "유제품",
        "기타 간식",
    ],
}


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _is_chestnut_trap(query_text: str) -> bool:
    """
    '야식' 같은 쿼리에서 밤(栗) 토큰이 엉뚱하게 상위로 뜨는 문제를 완화.
    아주 단순한 룰 기반으로만 감점 적용.
    """
    q = _normalize(query_text)

    # 야식/밤에 먹는 등 "night snack" 의도인데
    # 결과 음식명이 "밤식빵/밤밥/통밤..." 같은 밤로 오인되는 케이스 완화
    return ("야식" in q) or ("밤에" in q) or ("늦은" in q)


def apply_category_guardrails(
    hits: pd.DataFrame,
    *,
    kind: str,
    query_text: str,
    penalty_other: float = 0.15,
    bonus_match: float = 0.05,
    penalty_chestnut: float = 0.30,
    debug: bool = False,
) -> pd.DataFrame:
    """
    A안: drop(필터링)하지 않고 'score'만 조정한다.
    - kind가 기대하는 cat_large면 소폭 가점(bonus_match)
    - 아니면 감점(penalty_other)
    - '야식' 쿼리에서 밤(栗) 오인 토큰이 보이면 추가 감점(penalty_chestnut)

    입력 hits는 반드시 'score', 'cat_large', 'food_name' 컬럼을 가진다고 가정.
    """
    if hits is None or len(hits) == 0:
        return hits

    if "score" not in hits.columns:
        return hits

    allowed = set(KIND_ALLOWED_CAT_LARGE.get(kind, []))

    out = hits.copy()

    # 점수 조정: cat_large 일치/불일치
    if "cat_large" in out.columns and allowed:
        is_match = out["cat_large"].fillna("").isin(allowed)
        out.loc[is_match, "score"] = out.loc[is_match, "score"] + bonus_match
        out.loc[~is_match, "score"] = out.loc[~is_match, "score"] - penalty_other

    # 밤(栗) 트랩 감점(야식 의도에서만)
    if _is_chestnut_trap(query_text) and "food_name" in out.columns:
        # "밤" 단어가 들어간 결과만 감점(너무 과하면 regex를 더 빡세게 가능)
        has_bam = out["food_name"].fillna("").str.contains(r"밤", regex=True)
        out.loc[has_bam, "score"] = out.loc[has_bam, "score"] - penalty_chestnut

    # 정렬 유지: score 기준 내림차순
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    if debug:
        try:
            top = out["cat_large"].value_counts().head(5).to_dict()
            print(f"[DBG] guardrails kind={kind} allowed={len(allowed)} top_cat_large={top}")
        except Exception:
            pass

    return out
