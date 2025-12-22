from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Set

QueryKind = Literal["food", "drink", "snack"]

# food_data.csv의 cat_large 값 기준으로 묶기
CAT_DRINK: Set[str] = {
    "음료 및 차류",
}

CAT_SNACK: Set[str] = {
    "빵 및 과자류",
    "유제품류 및 빙과류",
}

# 나머지는 전부 food로 처리(식사/반찬/요리류)
def kind_of_cat_large(cat_large: str) -> QueryKind:
    if cat_large in CAT_DRINK:
        return "drink"
    if cat_large in CAT_SNACK:
        return "snack"
    return "food"
