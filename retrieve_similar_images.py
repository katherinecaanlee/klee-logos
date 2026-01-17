#!/usr/bin/env python3
"""
Retrieve the top-K similar logo PNGs from the pgvector database using a freeform text query.

Usage:
  python retrieve_similar_images.py "warm geometric coffee brand" --k 3
  python retrieve_similar_images.py "minimal industrial tech" --k 5 --save-dir ./samples

Requires .env with:
  SUPABASE_DB_URL=...
  LOGOBOOK_S3_BUCKET=...  (or S3_BUCKET)
  AWS_DEFAULT_REGION=...  (or AWS_REGION)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN (if needed)
"""

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List

import boto3
import psycopg2
import torch
import open_clip
from dotenv import load_dotenv


# -------------------
# Env / clients
# -------------------

load_dotenv(dotenv_path=".env")


def get_env(name: str, default: str = "", required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


SUPABASE_DB_URL = get_env("SUPABASE_DB_URL", required=True)
LOGOBOOK_S3_BUCKET = get_env("LOGOBOOK_S3_BUCKET") or get_env("S3_BUCKET", required=True)
AWS_REGION = get_env("AWS_DEFAULT_REGION") or get_env("AWS_REGION") or "us-east-1"

s3 = boto3.client("s3", region_name=AWS_REGION)
DEVICE = "cpu"


# -------------------
# OpenCLIP text encoder
# -------------------

def _load_openclip_text_encoder():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer


_OPENCLIP_MODEL, _OPENCLIP_TOKENIZER = _load_openclip_text_encoder()


def embed_text_openclip(text: str) -> List[float]:
    with torch.no_grad():
        tokens = _OPENCLIP_TOKENIZER([text]).to(DEVICE)
        feats = _OPENCLIP_MODEL.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats[0].cpu().numpy().astype("float32").tolist()
        return vec


# -------------------
# pgvector retrieval
# -------------------

def _pg_connect():
    return psycopg2.connect(SUPABASE_DB_URL)


def _vec_to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def retrieve_top_k(query_text: str, k: int = 3) -> List[Dict[str, Any]]:
    query_vec = embed_text_openclip(query_text)
    vec_lit = _vec_to_pgvector_literal(query_vec)

    sql = """
        SELECT asset_id, category, class, s3_png_key,
               (embedding <=> %s::vector) AS distance
        FROM logo_assets
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [vec_lit, vec_lit, k]

    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            # Fallback: ivfflat + cosine index returns 0 rows on this DB.
            # Force a sequential scan if that happens.
            if not rows:
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")
                cur.execute("SET LOCAL enable_indexonlyscan = off;")
                cur.execute(sql, params)
                rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for asset_id, category, cls, s3_png_key, distance in rows:
        results.append(
            {
                "asset_id": str(asset_id),
                "category": str(category),
                "class": str(cls),
                "s3_png_key": str(s3_png_key),
                "distance": float(distance),
            }
        )
    return results


# -------------------
# S3 helpers
# -------------------

def download_png(bucket: str, key: str, out_dir: pathlib.Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = key.rsplit("/", 1)[-1] or "logo.png"
    path = out_dir / filename
    obj = s3.get_object(Bucket=bucket, Key=key)
    path.write_bytes(obj["Body"].read())
    return str(path)


# -------------------
# CLI
# -------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-K similar logos from pgvector via a text query.")
    parser.add_argument("query", help="A sentence or keywords describing the desired style/feel.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return (default 3).")
    parser.add_argument("--save-dir", type=str, help="Optional directory to download the PNGs.")
    args = parser.parse_args()

    results = retrieve_top_k(args.query, k=args.k)

    if args.save_dir:
        saved = []
        out_dir = pathlib.Path(args.save_dir)
        for r in results:
            path = download_png(LOGOBOOK_S3_BUCKET, r["s3_png_key"], out_dir)
            saved.append({**r, "local_file": path})
        print(json.dumps({"results": saved}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
