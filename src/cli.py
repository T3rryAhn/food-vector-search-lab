from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from .config import settings
from .load_data import load_food_df
from .embedder import LocalSentenceTransformerEmbedder
from .text_index import build_faiss_index, load_index, load_meta, search_index
from .recommender import NutrientFilters, NutrientWeights, apply_filters, score_by_nutrients, hybrid_rank
from .eval_queries import EVAL_QUERIES

MODEL_MAP = {
    "snowflake": settings.model_snowflake,
    "jina": settings.model_jina,
}

def resolve_model_name(model_key: str) -> str:
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(MODEL_MAP.keys())}")
    return MODEL_MAP[model_key]

def index_name_for(model_key: str) -> str:
    return f"food_{model_key}"

def index_paths(index_name: str) -> tuple[Path, Path]:
    index_path = settings.indices_dir / f"{index_name}.faiss"
    meta_path = settings.indices_dir / f"{index_name}.parquet"
    return index_path, meta_path

def cmd_build_index(args):
    df = load_food_df(args.data)
    model_name = resolve_model_name(args.model)

    embedder = LocalSentenceTransformerEmbedder(model_name=model_name, device=args.device)

    idx_name = args.index_name or index_name_for(args.model)
    art = build_faiss_index(
        df=df,
        embedder=embedder,
        indices_dir=settings.indices_dir,
        index_name=idx_name,
        batch_size=args.batch_size,
    )
    print(f"[OK] model     : {args.model} ({model_name})")
    print(f"[OK] index     : {art.index_path}")
    print(f"[OK] meta      : {art.meta_path}")
    print(f"[OK] dim       : {art.dim}")

def cmd_search(args):
    model_name = resolve_model_name(args.model)
    embedder = LocalSentenceTransformerEmbedder(model_name=model_name, device=args.device)

    idx_name = args.index_name or index_name_for(args.model)
    index_path, meta_path = index_paths(idx_name)

    index = load_index(index_path)
    meta = load_meta(meta_path)

    hits = search_index(index, embedder, meta, args.q, topk=args.topk)
    print(hits[[
        "score", "food_name", "cat_large", "cat_mid",
        "kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","caffeine_mg"
    ]].to_string(index=False))

def cmd_recommend(args):
    idx_name = args.index_name or index_name_for(args.model)
    _, meta_path = index_paths(idx_name)
    df = load_meta(meta_path)

    f = NutrientFilters(
        protein_min=args.protein_min,
        sugar_max=args.sugar_max,
        sodium_max=args.sodium_max,
        caffeine_max=args.caffeine_max,
        kcal_max=args.kcal_max,
    )
    w = NutrientWeights(
        protein_g=args.w_protein,
        sugar_g=args.w_sugar,
        sodium_mg=args.w_sodium,
        kcal=args.w_kcal,
        fat_g=args.w_fat,
        carb_g=args.w_carb,
        caffeine_mg=args.w_caffeine,
    )

    cand = apply_filters(df, f)
    ranked = score_by_nutrients(cand, w).head(args.topk)

    print(ranked[[
        "nutri_score", "food_name", "cat_large", "cat_mid",
        "kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","caffeine_mg"
    ]].to_string(index=False))

def cmd_hybrid(args):
    model_name = resolve_model_name(args.model)
    embedder = LocalSentenceTransformerEmbedder(model_name=model_name, device=args.device)

    idx_name = args.index_name or index_name_for(args.model)
    index_path, meta_path = index_paths(idx_name)

    index = load_index(index_path)
    meta = load_meta(meta_path)

    hits = search_index(index, embedder, meta, args.q, topk=args.pool)

    f = NutrientFilters(
        protein_min=args.protein_min,
        sugar_max=args.sugar_max,
        sodium_max=args.sodium_max,
        caffeine_max=args.caffeine_max,
        kcal_max=args.kcal_max,
    )
    hits2 = apply_filters(hits, f)

    w = NutrientWeights(
        protein_g=args.w_protein,
        sugar_g=args.w_sugar,
        sodium_mg=args.w_sodium,
        kcal=args.w_kcal,
        fat_g=args.w_fat,
        carb_g=args.w_carb,
        caffeine_mg=args.w_caffeine,
    )

    ranked = hybrid_rank(hits2, w, alpha=args.alpha).head(args.topk)
    print(ranked[[
        "final_score", "score", "nutri_score", "food_name", "cat_large", "cat_mid",
        "kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","caffeine_mg"
    ]].to_string(index=False))

def cmd_compare(args):
    """
    동일 쿼리셋(EVAL_QUERIES)을 두 모델로 검색해 Top-K를 저장.
    - 정량 점수는 아직(라벨이 없으니) 넣지 않고, 비교 테이블 + 케이스 스터디를 뽑는다.
    """
    topk = args.topk
    models = args.models

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = settings.reports_dir / f"compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 인덱스/메타 로드 (모델별 1회)
    runners = {}
    for mk in models:
        model_name = resolve_model_name(mk)
        embedder = LocalSentenceTransformerEmbedder(model_name=model_name, device=args.device)

        idx_name = index_name_for(mk)
        index_path, meta_path = index_paths(idx_name)

        index = load_index(index_path)
        meta = load_meta(meta_path)

        runners[mk] = {"model_name": model_name, "embedder": embedder, "index": index, "meta": meta}

    rows = []
    for item in EVAL_QUERIES:
        q = item["q"]
        notes = item.get("notes", "")

        for mk in models:
            r = runners[mk]
            hits = search_index(r["index"], r["embedder"], r["meta"], q, topk=topk)

            # Top1~TopK 음식명만 빠르게 비교하기 위해 리스트로 저장
            top_names = hits["food_name"].tolist()
            rows.append({
                "query": q,
                "notes": notes,
                "model_key": mk,
                "model_name": r["model_name"],
                "topk": topk,
                "top_foods": top_names,
            })

            # 모델별 상세 결과도 파일로 저장
            hits_out = out_dir / f"hits_{mk}_{safe_filename(q)}.csv"
            hits.to_csv(hits_out, index=False, encoding="utf-8-sig")

    # 비교 요약 저장
    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] compare report saved: {out_dir}")
    print(f" - {summary_csv.name}")
    print(f" - {summary_json.name}")
    print("Tip: open per-query CSVs in outputs/reports/... to inspect differences quickly.")

def safe_filename(s: str) -> str:
    # 파일명 안전 문자만 남김
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "_", "-", "+"):
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:60] if len(out) > 60 else out

def main():
    p = argparse.ArgumentParser(prog="food-vector-search-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index")
    b.add_argument("--data", default=str(settings.data_path))
    b.add_argument("--model", choices=["snowflake", "jina"], default="snowflake")
    b.add_argument("--index-name", default=None)
    b.add_argument("--batch-size", type=int, default=128)
    b.add_argument("--device", default=None, help='e.g. "cpu" or "cuda"')
    b.set_defaults(func=cmd_build_index)

    s = sub.add_parser("search")
    s.add_argument("--model", choices=["snowflake", "jina"], default="snowflake")
    s.add_argument("--index-name", default=None)
    s.add_argument("--q", required=True)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--device", default=None, help='e.g. "cpu" or "cuda"')
    s.set_defaults(func=cmd_search)

    r = sub.add_parser("recommend")
    r.add_argument("--model", choices=["snowflake", "jina"], default="snowflake")
    r.add_argument("--index-name", default=None)
    r.add_argument("--topk", type=int, default=10)

    r.add_argument("--protein-min", type=float, default=None)
    r.add_argument("--sugar-max", type=float, default=None)
    r.add_argument("--sodium-max", type=float, default=None)
    r.add_argument("--caffeine-max", type=float, default=None)
    r.add_argument("--kcal-max", type=float, default=None)

    r.add_argument("--w-protein", type=float, default=1.0)
    r.add_argument("--w-sugar", type=float, default=-0.5)
    r.add_argument("--w-sodium", type=float, default=-0.3)
    r.add_argument("--w-kcal", type=float, default=0.0)
    r.add_argument("--w-fat", type=float, default=0.0)
    r.add_argument("--w-carb", type=float, default=0.0)
    r.add_argument("--w-caffeine", type=float, default=0.0)
    r.set_defaults(func=cmd_recommend)

    h = sub.add_parser("hybrid")
    h.add_argument("--model", choices=["snowflake", "jina"], default="snowflake")
    h.add_argument("--index-name", default=None)
    h.add_argument("--q", required=True)
    h.add_argument("--pool", type=int, default=50)
    h.add_argument("--topk", type=int, default=10)
    h.add_argument("--alpha", type=float, default=0.7)
    h.add_argument("--device", default=None, help='e.g. "cpu" or "cuda"')

    h.add_argument("--protein-min", type=float, default=None)
    h.add_argument("--sugar-max", type=float, default=None)
    h.add_argument("--sodium-max", type=float, default=None)
    h.add_argument("--caffeine-max", type=float, default=None)
    h.add_argument("--kcal-max", type=float, default=None)

    h.add_argument("--w-protein", type=float, default=1.0)
    h.add_argument("--w-sugar", type=float, default=-0.5)
    h.add_argument("--w-sodium", type=float, default=-0.3)
    h.add_argument("--w-kcal", type=float, default=0.0)
    h.add_argument("--w-fat", type=float, default=0.0)
    h.add_argument("--w-carb", type=float, default=0.0)
    h.add_argument("--w-caffeine", type=float, default=0.0)
    h.set_defaults(func=cmd_hybrid)

    c = sub.add_parser("compare")
    c.add_argument("--models", nargs="+", default=["snowflake", "jina"], choices=["snowflake", "jina"])
    c.add_argument("--topk", type=int, default=10)
    c.add_argument("--device", default=None, help='e.g. "cpu" or "cuda"')
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
