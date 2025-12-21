from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import faiss

from .load_data import canonical_text_row

@dataclass
class IndexArtifacts:
    index_path: Path
    meta_path: Path
    dim: int

def build_faiss_index(
    df: pd.DataFrame,
    embedder,
    indices_dir: Path,
    index_name: str,
    batch_size: int = 128,
) -> IndexArtifacts:
    texts = [canonical_text_row(row) for _, row in df.iterrows()]
    vectors = embedder.embed(texts, batch_size=batch_size)

    dim = vectors.shape[1]
    # cosine 유사도: normalize_embeddings=True + inner product
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = indices_dir / f"{index_name}.faiss"
    meta_path = indices_dir / f"{index_name}.parquet"

    faiss.write_index(index, str(index_path))

    meta = df[[
        "food_code", "food_name", "cat_large", "cat_mid", "cat_small", "cat_detail",
        "kcal", "carb_g", "protein_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"
    ]].copy()
    meta.to_parquet(meta_path, index=False)

    return IndexArtifacts(index_path=index_path, meta_path=meta_path, dim=dim)

def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))

def load_meta(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")
    return pd.read_parquet(meta_path)

def search_index(index: faiss.Index, embedder, meta_df: pd.DataFrame, query: str, topk: int = 10) -> pd.DataFrame:
    qvec = embedder.embed([query], batch_size=1)
    scores, idxs = index.search(qvec, topk)

    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    hits = meta_df.iloc[idxs].copy()
    hits.insert(0, "score", scores)
    return hits
