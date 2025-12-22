import pandas as pd
import numpy as np

NUT_COLS = ["kcal", "carb_g", "protein_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"]

def load_food_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"food_code", "food_name", "cat_large", "cat_mid", "cat_small", "cat_detail"} | set(NUT_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # 영양소 숫자화
    for c in NUT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 텍스트 컬럼 정리
    for c in ["cat_large", "cat_mid", "cat_small", "cat_detail"]:
        df[c] = df[c].fillna("").astype(str)

    df["food_name"] = df["food_name"].fillna("").astype(str)
    df["food_code"] = df["food_code"].fillna("").astype(str)

    return df

def canonical_text_row(row) -> str:
    """
    임베딩용 대표 텍스트.
    모델이 '상식으로' 보완할 수 있도록 카테고리 + 영양 요약을 같이 제공.
    """
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "unknown"
        return f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)

    cat = "/".join([row["cat_large"], row["cat_mid"], row["cat_small"], row["cat_detail"]]).strip("/")
    nutrients = (
        f"kcal {fmt(row['kcal'])}, carb {fmt(row['carb_g'])}g, "
        f"protein {fmt(row['protein_g'])}g, fat {fmt(row['fat_g'])}g, "
        f"sugar {fmt(row['sugar_g'])}g, sodium {fmt(row['sodium_mg'])}mg, "
        f"caffeine {fmt(row['caffeine_mg'])}mg"
    )
    return f"{row['food_name']} | category: {cat} | nutrition(per100): {nutrients}"
