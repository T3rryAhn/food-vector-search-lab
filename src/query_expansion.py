"""
짧고 추상적인 쿼리를 의미가 풍부한 문장으로 확장한다.
MVP 단계에서는 규칙 기반(hard-coded) 방식 사용.
"""

QUERY_EXPANSION_MAP = {
    "야식": "야식 늦은 밤에 먹는 간단한 음식 간식 부담 없는 음식",
    "시원한": "시원한 차가운 아이스 iced cold",
    "차가운": "차가운 시원한 아이스 iced cold",
    "따뜻한": "따뜻한 뜨거운 hot warm",
    "뜨거운": "뜨거운 따뜻한 hot warm",
    "매콤한": "매콤한 매운 spicy",
    "순한": "순한 매운맛이 없는 자극적이지 않은 mild",
    "달콤한": "달콤한 단 sweet",
    "짭짤한": "짭짤한 짠 salty",
    "가벼운": "가벼운 가볍게 먹을 수 있는 light",
    "든든한": "든든한 배부른 hearty filling",
    "편한": "편한 편안한 comfortable",
    "무거운": "무거운 무겁게 먹을 수 있는 heavy",
}

def expand_query(query: str) -> str:
    q = query.strip()
    additions = []
    for key, aug in QUERY_EXPANSION_MAP.items():
        if key and key in q:
            additions.append(aug)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for a in additions:
        if a not in seen:
            uniq.append(a)
            seen.add(a)

    if uniq:
        return f"{q} {' '.join(uniq)}".strip()
    return q
