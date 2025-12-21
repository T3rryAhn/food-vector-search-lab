"""
검색 결과 다양화 후처리.
- 동일 cat_mid 반복 제한
- 음식명 핵심 토큰 반복 제한
- topk가 부족하면 raw hits로 채우는 fallback 제공
"""

from __future__ import annotations
import re
from typing import Optional
import pandas as pd

def _main_token(food_name: str) -> str:
    if not isinstance(food_name, str) or not food_name:
        return ""
    tokens = re.split(r"[ _\-]+", food_name.strip())
    return tokens[0] if tokens else food_name.strip()

def diversify_results(
    df: pd.DataFrame,
    topk: int = 10,
    max_same_cat_mid: int = 2,
    max_same_token: int = 2,
    fallback: bool = True,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    picked_rows = []
    cat_count = {}
    token_count = {}
    picked_idx = set()

    for idx, row in df.iterrows():
        cat = row.get("cat_mid", "")
        token = _main_token(row.get("food_name", ""))

        if cat_count.get(cat, 0) >= max_same_cat_mid:
            continue
        if token_count.get(token, 0) >= max_same_token:
            continue

        picked_rows.append(row)
        picked_idx.add(idx)

        cat_count[cat] = cat_count.get(cat, 0) + 1
        token_count[token] = token_count.get(token, 0) + 1

        if len(picked_rows) >= topk:
            break

    # fallback: topk 미달이면 원본 df에서 순서대로 채움(중복 제외)
    if fallback and len(picked_rows) < topk:
        for idx, row in df.iterrows():
            if idx in picked_idx:
                continue
            picked_rows.append(row)
            picked_idx.add(idx)
            if len(picked_rows) >= topk:
                break

    return pd.DataFrame(picked_rows).reset_index(drop=True)
