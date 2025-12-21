from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class NutrientFilters:
    kcal_min: float | None = None
    kcal_max: float | None = None
    carb_min: float | None = None
    carb_max: float | None = None
    protein_min: float | None = None
    protein_max: float | None = None
    fat_min: float | None = None
    fat_max: float | None = None
    sugar_max: float | None = None
    sodium_max: float | None = None
    caffeine_max: float | None = None

def apply_filters(df: pd.DataFrame, f: NutrientFilters) -> pd.DataFrame:
    q = df.copy()

    def cond(col, op, val):
        nonlocal q
        if val is None:
            return
        if op == ">=":
            q = q[q[col].notna() & (q[col] >= val)]
        elif op == "<=":
            q = q[q[col].notna() & (q[col] <= val)]

    cond("kcal", ">=", f.kcal_min); cond("kcal", "<=", f.kcal_max)
    cond("carb_g", ">=", f.carb_min); cond("carb_g", "<=", f.carb_max)
    cond("protein_g", ">=", f.protein_min); cond("protein_g", "<=", f.protein_max)
    cond("fat_g", ">=", f.fat_min); cond("fat_g", "<=", f.fat_max)
    cond("sugar_g", "<=", f.sugar_max)
    cond("sodium_mg", "<=", f.sodium_max)
    cond("caffeine_mg", "<=", f.caffeine_max)

    return q

@dataclass
class NutrientWeights:
    # 높을수록 선호(+) / 낮을수록 선호(-)를 스코어로 반영
    kcal: float = 0.0
    carb_g: float = 0.0
    protein_g: float = 1.0
    fat_g: float = 0.0
    sugar_g: float = -0.5
    sodium_mg: float = -0.3
    caffeine_mg: float = 0.0

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd

def score_by_nutrients(df: pd.DataFrame, w: NutrientWeights) -> pd.DataFrame:
    q = df.copy()

    # z-score로 스케일 통일
    cols = ["kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","caffeine_mg"]
    for c in cols:
        q[f"z_{c}"] = zscore(q[c].fillna(q[c].mean()))

    q["nutri_score"] = (
        q["z_kcal"] * w.kcal +
        q["z_carb_g"] * w.carb_g +
        q["z_protein_g"] * w.protein_g +
        q["z_fat_g"] * w.fat_g +
        q["z_sugar_g"] * w.sugar_g +
        q["z_sodium_mg"] * w.sodium_mg +
        q["z_caffeine_mg"] * w.caffeine_mg
    )

    return q.sort_values("nutri_score", ascending=False)

def hybrid_rank(text_hits: pd.DataFrame, w: NutrientWeights, alpha: float = 0.7) -> pd.DataFrame:
    """
    text_hits: score(코사인 유사도) 포함
    alpha: 텍스트 점수 비중(0~1)
    """
    q = text_hits.copy()

    # 텍스트 score / 영양 score 둘 다 표준화 후 합산
    q = score_by_nutrients(q, w)

    q["z_text"] = zscore(q["score"].fillna(q["score"].mean()))
    q["z_nutri"] = zscore(q["nutri_score"].fillna(q["nutri_score"].mean()))
    q["final_score"] = alpha * q["z_text"] + (1 - alpha) * q["z_nutri"]

    return q.sort_values("final_score", ascending=False)
