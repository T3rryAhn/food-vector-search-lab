from fastapi import HTTPException, Query
from typing import Optional
import pandas as pd
from src.query_expansion import expand_query
from src.diversify import diversify_results
resources = {}

def recommend_food(
    query: str = Query(..., min_length=1, max_length=240),
    # 카테고리 필터 (예: '면', '밥', '국', '디저트')
    category: Optional[str] = Query(None, description="원하는 카테고리 (포함 검색)"),
    k: int = Query(5, ge=1, le=10)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    # 1. 의도 파악 (Food/Drink/Snack)
    intent_kind = "food"
    if any(w in query for w in ["커피", "라떼", "아메리카노", "주스", "에이드", "티", "차"]):
        intent_kind = "drink"
    elif any(w in query for w in ["케이크", "빵", "디저트", "과자", "아이스크림"]):
        intent_kind = "snack"

    # 2. 쿼리 확장
    expanded_query = expand_query(query, kind=intent_kind)
    # Do not persist user queries.

    # 3. 검색
    search_text = f"Represent this sentence for searching relevant passages: {expanded_query}"
    query_vector = resources['model'].encode([search_text], normalize_embeddings=True)

    # 카테고리 필터가 있으면 필터링 되어서 버려지는 게 많으므로,
    # 후보를 아주 넉넉하게(500개)
    fetch_k = 500
    distances, indices = resources['index'].search(query_vector, fetch_k)

    # 4. 결과 취합 및 필터링
    temp_results = []
    seen_names = set()

    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i]

        if idx < 0 or idx >= len(resources['metadata']): continue
        item = resources['metadata'][idx]

        # 카테고리 필터링 로직
        # 사용자가 '면'을 입력하면 -> '면 및 만두류' 포함되므로 통과
        item_cat = item['cat_large']
        if category and (category not in item_cat):
            continue

        # Guardrail: 의도와 다른 종류 감점 (카테고리 지정 안했을 때만 중요)
        final_score = float(score)
        if not category and item.get('kind', 'food') != intent_kind:
            final_score -= 0.2

        # 중복 제거
        if item['food_name'] in seen_names:
            continue
        seen_names.add(item['food_name'])

        temp_results.append({
            "food_code": item['food_code'],
            "food_name": item['food_name'],
            "cat_mid": item.get('cat_mid', ''),
            "category": item_cat,
            "tags": item.get('tags', ''),
            "score": final_score
        })

        # 필터링 후 k개가 채워지면 중단하지 않고, 다양화를 위해 3배수 정도 모음
        if len(temp_results) >= k * 3:
            break

    # 5. 다양화
    df_results = pd.DataFrame(temp_results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="score", ascending=False)
        df_diversified = diversify_results(
            df_results,
            topk=k,
            max_same_cat_mid=2,
            max_same_token=2
        )
        return df_diversified.to_dict(orient="records")

    return []
