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
from .recommender import (
    NutrientFilters,
    NutrientWeights,
    apply_filters,
    score_by_nutrients,
    hybrid_rank,
)
from .eval_queries import EVAL_QUERIES
from .query_expansion import expand_query
from .diversify import diversify_results
from .post_filter import apply_category_guardrails

MODEL_MAP = {
    "snowflake": settings.model_snowflake,
    "jina": settings.model_jina,
}


def resolve_model_name(model_key: str) -> str:
    if model_key not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from {list(MODEL_MAP.keys())}"
        )
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

    embedder = LocalSentenceTransformerEmbedder(
        model_name=model_name, device=args.device
    )

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
    embedder = LocalSentenceTransformerEmbedder(
        model_name=model_name, device=args.device
    )

    idx_name = args.index_name or index_name_for(args.model)
    index_path, meta_path = index_paths(idx_name)

    index = load_index(index_path)
    meta = load_meta(meta_path)

    # 쿼리 확장
    q = expand_query(args.q)

    # 먼저 넉넉히 뽑고(raw_topk), 그 다음 다양화로 topk 구성
    raw_topk = max(args.topk * 10, args.topk)  # 기본: 10배 후보 풀
    hits_raw = search_index(index, embedder, meta, q, topk=raw_topk)

    # cat_large 기반 가드레일 적용
    hits_raw = apply_category_guardrails(hits_raw, kind=args.kind, query_text=args.q)

    hits = diversify_results(
        hits_raw,
        topk=args.topk,
        max_same_cat_mid=args.max_same_cat_mid,
        max_same_token=args.max_same_token,
    )

    print(
        hits[
            [
                "score",
                "food_name",
                "cat_large",
                "cat_mid",
                "kcal",
                "carb_g",
                "protein_g",
                "fat_g",
                "sugar_g",
                "sodium_mg",
                "caffeine_mg",
            ]
        ].to_string(index=False)
    )


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
    ranked = score_by_nutrients(cand, w)

    # 다양화 적용
    if args.diversify:
        ranked = diversify_results(
            ranked,
            topk=args.topk,
            max_same_cat_mid=args.max_same_cat_mid,
            max_same_token=args.max_same_token,
        )
    else:
        ranked = ranked.head(args.topk)

    print(
        ranked[
            [
                "nutri_score",
                "food_name",
                "cat_large",
                "cat_mid",
                "kcal",
                "carb_g",
                "protein_g",
                "fat_g",
                "sugar_g",
                "sodium_mg",
                "caffeine_mg",
            ]
        ].to_string(index=False)
    )


def cmd_hybrid(args):
    model_name = resolve_model_name(args.model)
    embedder = LocalSentenceTransformerEmbedder(
        model_name=model_name, device=args.device
    )

    idx_name = args.index_name or index_name_for(args.model)
    index_path, meta_path = index_paths(idx_name)

    index = load_index(index_path)
    meta = load_meta(meta_path)

    # 쿼리 확장
    q = expand_query(args.q)

    # 후보 풀(pool) 확보 후, 다양화로 도배 감소
    hits_raw = search_index(index, embedder, meta, q, topk=args.pool)

    hits_raw = apply_category_guardrails(hits_raw, kind=args.kind, query_text=args.q)

    hits_div = diversify_results(
        hits_raw,
        topk=args.pool_diversified,  # 다양화 후 남길 후보 수(재랭킹 후보 풀)
        max_same_cat_mid=args.max_same_cat_mid,
        max_same_token=args.max_same_token,
    )

    # 영양소 필터
    f = NutrientFilters(
        protein_min=args.protein_min,
        sugar_max=args.sugar_max,
        sodium_max=args.sodium_max,
        caffeine_max=args.caffeine_max,
        kcal_max=args.kcal_max,
    )
    hits2 = apply_filters(hits_div, f)

    # 4) 영양소 재랭킹 + 텍스트 결합
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
    print(
        ranked[
            [
                "final_score",
                "score",
                "nutri_score",
                "food_name",
                "cat_large",
                "cat_mid",
                "kcal",
                "carb_g",
                "protein_g",
                "fat_g",
                "sugar_g",
                "sodium_mg",
                "caffeine_mg",
            ]
        ].to_string(index=False)
    )


def cmd_compare(args):
    """
    동일 쿼리셋(EVAL_QUERIES)을 두 모델로 검색해 Top-K를 저장.
    - compare에서도 쿼리 확장 + 다양화 적용 가능하게 한다.
    - raw hits와 diversified hits를 둘 다 저장해, 개선 효과를 나중에 비교하기 쉽도록 한다.
    """
    topk = args.topk
    models = args.models

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = settings.reports_dir / f"compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = {}
    for mk in models:
        model_name = resolve_model_name(mk)
        embedder = LocalSentenceTransformerEmbedder(
            model_name=model_name, device=args.device
        )

        idx_name = index_name_for(mk)
        index_path, meta_path = index_paths(idx_name)

        index = load_index(index_path)
        meta = load_meta(meta_path)

        runners[mk] = {
            "model_name": model_name,
            "embedder": embedder,
            "index": index,
            "meta": meta,
        }

    rows = []
    for item in EVAL_QUERIES:
        q0 = item["q"]
        notes = item.get("notes", "")
        kind = item.get("kind", "food")

        q = expand_query(q0) if args.use_query_expansion else q0

        for mk in models:
            r = runners[mk]

            raw_topk = max(topk * args.raw_multiplier, topk)
            hits_raw = search_index(
                r["index"], r["embedder"], r["meta"], q, topk=raw_topk
            )

            if args.use_guardrails:
                hits_raw = apply_category_guardrails(
                    hits_raw,
                    kind=kind,
                    query_text=q0,
                    penalty_other=args.penalty_other,
                    bonus_match=args.bonus_match,
                    penalty_chestnut=args.penalty_chestnut,
                )

            if args.use_diversify:
                hits = diversify_results(
                    hits_raw,
                    topk=topk,
                    max_same_cat_mid=args.max_same_cat_mid,
                    max_same_token=args.max_same_token,
                )
            else:
                hits = hits_raw.head(topk)

            # 요약 저장(Top1~TopK 음식명)
            top_names = hits["food_name"].tolist()
            rows.append(
                {
                    "query_original": q0,
                    "query_used": q,
                    "notes": notes,
                    "kind": kind,
                    "model_key": mk,
                    "model_name": r["model_name"],
                    "topk": topk,
                    "top_foods": top_names,
                    "use_query_expansion": args.use_query_expansion,
                    "use_diversify": args.use_diversify,
                }
            )

            # raw / diversified 둘 다 저장
            hits_raw_out = out_dir / f"hits_raw_{mk}_{safe_filename(q0)}.csv"
            hits_out = out_dir / f"hits_{mk}_{safe_filename(q0)}.csv"

            hits_raw.to_csv(hits_raw_out, index=False, encoding="utf-8-sig")
            hits.to_csv(hits_out, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] compare report saved: {out_dir}")
    print(f" - {summary_csv.name}")
    print(f" - {summary_json.name}")
    print(
        "Tip: open per-query CSVs in outputs/reports/... to inspect differences quickly."
    )


def safe_filename(s: str) -> str:
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
    # 다양화 옵션
    s.add_argument("--max-same-cat-mid", type=int, default=2)
    s.add_argument("--max-same-token", type=int, default=2)
    s.set_defaults(func=cmd_search)

    s.add_argument("--kind", choices=["food","drink","snack"], required=True)

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

    r.add_argument("--diversify", action="store_true")
    r.add_argument("--max-same-cat-mid", type=int, default=2)
    r.add_argument("--max-same-token", type=int, default=2)
    r.set_defaults(func=cmd_recommend)

    h = sub.add_parser("hybrid")
    h.add_argument("--model", choices=["snowflake", "jina"], default="snowflake")
    h.add_argument("--index-name", default=None)
    h.add_argument("--q", required=True)
    h.add_argument("--pool", type=int, default=80)
    # 다양화 후 남길 후보 수(재랭킹 입력)
    h.add_argument("--pool-diversified", type=int, default=60)
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

    # 다양화 옵션
    h.add_argument("--max-same-cat-mid", type=int, default=2)
    h.add_argument("--max-same-token", type=int, default=2)
    h.set_defaults(func=cmd_hybrid)

    h.add_argument("--kind", choices=["food","drink","snack"], required=True)

    c = sub.add_parser("compare")
    c.add_argument(
        "--models",
        nargs="+",
        default=["snowflake", "jina"],
        choices=["snowflake", "jina"],
    )
    c.add_argument("--topk", type=int, default=10)
    c.add_argument("--device", default=None, help='e.g. "cpu" or "cuda"')
    # compare에서 확장/다양화 on/off
    c.add_argument("--use-query-expansion", action="store_true")
    c.add_argument("--use-diversify", action="store_true")
    c.add_argument(
        "--raw-multiplier",
        type=int,
        default=10,
        help="raw 후보를 topk의 몇 배로 뽑을지",
    )
    c.add_argument("--max-same-cat-mid", type=int, default=2)
    c.add_argument("--max-same-token", type=int, default=2)
    c.set_defaults(func=cmd_compare)

    c.add_argument("--use-guardrails", action="store_true", help="apply cat_large-based guardrails by kind")
    c.add_argument("--penalty-other", type=float, default=0.25)
    c.add_argument("--bonus-match", type=float, default=0.05)
    c.add_argument("--penalty-chestnut", type=float, default=0.35)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
