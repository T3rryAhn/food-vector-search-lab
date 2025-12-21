"""
검색 결과 다양화 후처리.
- 동일 cat_mid 반복 제한
- 음식명 핵심 토큰 반복 제한
"""

import re
import pandas as pd

def _main_token(food_name: str) -> str:
    """
    음식명에서 반복 판단용 핵심 토큰 추출.
    (아주 단순한 MVP 버전)
    """
    if not isinstance(food_name, str):
        return ""
    # 구분자로 나누고 첫 번째 토큰 반환
    tokens = re.split(r"[ _\-]", food_name)
    return tokens[0] if tokens else food_name

def diversify_results(
    df: pd.DataFrame,
    topk: int = 10,
    max_same_cat_mid: int = 2,
    max_same_token: int = 2,
):
    results = []
    cat_count = {}
    token_count = {}

    for _, row in df.iterrows():
        cat = row.get("cat_mid", "")
        food_name = row.get("food_name", "")
        token = _main_token(food_name)

        # 1. 카테고리 중복 체크
        if cat and cat_count.get(cat, 0) >= max_same_cat_mid:
            continue
        
        # 2. 핵심 토큰(음식명 유사도) 중복 체크
        if token and token_count.get(token, 0) >= max_same_token:
            continue

        results.append(row)
        
        # 카운트 업데이트
        cat_count[cat] = cat_count.get(cat, 0) + 1
        token_count[token] = token_count.get(token, 0) + 1

        if len(results) >= topk:
            break

    if not results:
        return df.iloc[0:0]
    
    # 리스트에 담긴 Series들을 DataFrame으로 변환 후 연결
    return pd.concat([pd.DataFrame([res]) for res in results], ignore_index=True)