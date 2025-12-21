# food_vector_search_lab/post_filter.py
from __future__ import annotations

import re
import pandas as pd

# food_data.csv의 cat_large 분포 기준
CAT_LARGE_DRINK = {"음료 및 차류"}
CAT_LARGE_SNACK = {"빵 및 과자류", "유제품류 및 빙과류"}

# food(kind=food)에서 우선적으로 살릴 “식사류” 카테고리(국물/한끼 의도에 특히 중요)
CAT_LARGE_MEAL = {
    "밥류",
    "국 및 탕류",
    "찌개 및 전골류",
    "면 및 만두류",
    "죽 및 스프류",
    "볶음류",
    "구이류",
    "튀김류",
    "찜류",
    "조림류",
    "전·적 및 부침류",
    "나물·숙채류",
    "생채·무침류",
    "김치류",
    # 나머지는 “기본 food”로 허용
}

# 쿼리 키워드 -> cat_large 부스팅(“추상 키워드”를 데이터 카테고리로 번역)
INTENT_BOOST_MAP: list[tuple[re.Pattern, dict[str, float]]] = [
    # 국물/따뜻한 식사류
    (
        re.compile(r"(국물|국\b|탕\b|찌개|전골|해장|수프|스프)"),
        {
            "국 및 탕류": 0.55,
            "찌개 및 전골류": 0.55,
            "죽 및 스프류": 0.35,
            "면 및 만두류": 0.20,
        },
    ),
    # 음료 의도
    (
        re.compile(r"(음료|차\b|커피|라떼|주스|티\b|스무디|에이드)"),
        {
            "음료 및 차류": 0.55,
        },
    ),
    # 간식/디저트 의도
    (
        re.compile(r"(간식|디저트|빵|과자|케이크|쿠키|아이스크림|빙수|와플|도넛|초콜릿)"),
        {
            "빵 및 과자류": 0.45,
            "유제품류 및 빙과류": 0.45,
        },
    ),
]

# “야식”이 “밤(栗)”로 오인되는 케이스(밤식빵/밤밥 등) 감점: cat_large가 아니라 이름 토큰 기반
CHESTNUT_PATTERN = re.compile(r"(^|[^가-힣])밤([가-힣]|$)")  # '밤' 단독/접두 정도를 잡는 느낌으로


def _kind_allowed_cat_large(kind: str) -> set[str]:
    if kind == "drink":
        return set(CAT_LARGE_DRINK)
    if kind == "snack":
        return set(CAT_LARGE_SNACK)
    # kind == "food"
    # food는 음료/간식 제외
    return set(df_cat_large_food_default())


def df_cat_large_food_default() -> set[str]:
    # food 허용군: 식사류(대부분) + 기타(장류 등) 포함
    # drink/snack은 제외 대상으로 별도 패널티/필터에서 처리
    return CAT_LARGE_MEAL | {
        "장류, 양념류",
        "장아찌·절임류",
        "젓갈류",
        "수·조·어·육류",
        "곡류, 서류 제품",
        "과일류",
        "두류, 견과 및 종실류",
        "채소, 해조류",
    }


def apply_category_guardrails(
    hits: pd.DataFrame,
    kind: str,
    query_text: str,
    *,
    hard_filter_mismatch: bool = True,
    penalty_other: float = 1.0,
    bonus_match: float = 0.05,
    penalty_chestnut: float = 0.35, 
) -> pd.DataFrame:
    if hits is None or len(hits) == 0:
        return hits

    out = hits.copy()

    # --- kind 기반 1차 가드레일 ---
    if kind == "food":
        mismatch_mask = out["cat_large"].isin(
            {"음료 및 차류", "빵 및 과자류", "유제품류 및 빙과류"}
        )
    elif kind == "drink":
        mismatch_mask = out["cat_large"] != "음료 및 차류"
    elif kind == "snack":
        mismatch_mask = ~out["cat_large"].isin(
            {"빵 및 과자류", "유제품류 및 빙과류"}
        )
    else:
        mismatch_mask = pd.Series(False, index=out.index)

    if hard_filter_mismatch:
        out = out.loc[~mismatch_mask].copy()
    else:
        out.loc[mismatch_mask, "score"] -= penalty_other

    if len(out) == 0:
        return hits

    # --- kind 일치 보너스 ---
    if kind == "food":
        out.loc[
            ~out["cat_large"].isin(
                {"음료 및 차류", "빵 및 과자류", "유제품류 및 빙과류"}
            ),
            "score",
        ] += bonus_match
    elif kind == "drink":
        out.loc[out["cat_large"] == "음료 및 차류", "score"] += bonus_match
    elif kind == "snack":
        out.loc[
            out["cat_large"].isin({"빵 및 과자류", "유제품류 및 빙과류"}),
            "score",
        ] += bonus_match

    # --- 야식 + 밤(栗) 트랩 ---
    if "야식" in (query_text or ""):
        chestnut_mask = out["food_name"].str.contains("밤", na=False)
        out.loc[chestnut_mask, "score"] -= penalty_chestnut

    return out.sort_values("score", ascending=False).reset_index(drop=True)

