#!/usr/bin/env python3
"""
transcript_to_logos.py

Pipeline:

  transcript
    -> (LLM) visual intent + keywords
    -> (OpenCLIP text encoder) query embedding
    -> (Supabase pgvector) retrieve S3 PNG anchors per logo type
    -> (LLM vision) produce 3 concepts:
         - word_mark
         - pictorial_logo
         - abstract_icon
       each with:
         - image_prompt (for gpt-image-1)
         - rationale (1 sentence)
    -> (OpenAI Images) generate 3 raster logos
    -> (LLM vision) critique + optional prompt refinement
    -> final: 3 PNGs from gpt-image-1 + metadata

Env (.env):

  OPENAI_API_KEY=...
  SUPABASE_DB_URL=postgresql://...
  LOGOBOOK_S3_BUCKET=semiotic-logobook   # or S3_BUCKET
  AWS_DEFAULT_REGION=us-east-2           # or whatever region
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN (if not using profile)

Requires:

  pip install openai python-dotenv boto3 psycopg2-binary torch open_clip_torch pillow
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

import torch
import open_clip


# -------------------
# Env / clients
# -------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
LOGOBOOK_S3_BUCKET = os.getenv("LOGOBOOK_S3_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL in .env")
if not LOGOBOOK_S3_BUCKET:
    raise RuntimeError("Missing LOGOBOOK_S3_BUCKET or S3_BUCKET in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
s3 = boto3.client("s3", region_name=AWS_REGION)

DEVICE = "cpu"


# -------------------
# Data structures
# -------------------

@dataclass
class Anchor:
    asset_id: str
    category: str
    cls: str
    s3_png_key: str
    distance: float
    data_url: str = ""   # base64 data URL for vision model


@dataclass
class Concept:
    logo_type: str                 # word_mark | pictorial_logo | abstract_icon
    rationale: str                 # 1 sentence
    image_prompt: str              # prompt used for gpt-image-1
    anchors: List[Anchor]
    image_b64: Optional[str] = None  # final raster from gpt-image-1


# -------------------
# Helpers
# -------------------

def _strip_markdown_json(s: str) -> str:
    """
    Remove common markdown wrappers (```json ... ``` or bare ``` ... ```).
    """
    text = s.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence (may be ``` or ```json)
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_braced_json(s: str) -> str:
    """
    If the string contains extra prose, grab the first {...} or [...] block.
    """
    text = s.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _safe_json_loads(s: str) -> Any:
    cleaned = _strip_markdown_json(s)
    cleaned = _extract_braced_json(cleaned)
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Model returned invalid JSON:\n{s}") from e


def _b64_to_data_url_png(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def _s3_get_png_as_data_url(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    b64 = base64.b64encode(body).decode("utf-8")
    return _b64_to_data_url_png(b64)


# -------------------
# OpenCLIP text embedding (same model you used for ingestion)
# Basically converts text to a vector so we can do similarity search on logobook db
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
# pgvector retrieval (matches your logo_assets schema)
# -------------------

def _pg_connect():
    return psycopg2.connect(SUPABASE_DB_URL)


def _vec_to_pgvector_literal(vec: List[float]) -> str:
    # pgvector expects: '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def retrieve_anchors(query_vec: List[float], logo_type: str, k: int = 25) -> List[Anchor]:
    """
    Query Supabase Postgres (logo_assets) via pgvector.

    Table layout (per your screenshot):

      logo_assets(
        asset_id text primary key,
        category text,        -- e.g. 'business', 'shape', 'object', 'nature', 'letters'
        class    text,        -- e.g. 'education', 'hexagon', 'transport'
        s3_svg_key text,
        s3_png_key text,
        embedding_model text,
        embedding vector(512),
        ...
      )
    """

    # Map required logo_type -> categories you want to bias toward
    type_to_categories = {
        "word_mark":       ["letters", "numbers", "business"],
        "pictorial_logo":  ["object", "objects", "nature", "business"],
        "abstract_icon":   ["shape", "shapes", "symbol", "symbols", "business"],
    }
    cats = type_to_categories.get(logo_type, [])

    vec_lit = _vec_to_pgvector_literal(query_vec)

    base = """
        SELECT asset_id, category, class, s3_png_key,
               (embedding <=> %s::vector) AS distance
        FROM logo_assets
    """

    rows: List[Tuple[str, str, str, str, float]] = []

    if cats:
        sql = base + """
            WHERE category = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [vec_lit, cats, vec_lit, k]
    else:
        sql = base + """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [vec_lit, vec_lit, k]

    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    # Fallback unfiltered if the category filter was too strict
    if len(rows) < max(5, k // 4):
        sql2 = base + " ORDER BY embedding <=> %s::vector LIMIT %s"
        with _pg_connect() as conn:
            with conn.cursor() as cur:
                # base already has one %s placeholder for the distance expression; append the ORDER BY + LIMIT params
                cur.execute(sql2, [vec_lit, vec_lit, k])
                rows = cur.fetchall()

    anchors: List[Anchor] = []
    for asset_id, category, cls, s3_png_key, distance in rows:
        anchors.append(
            Anchor(
                asset_id=str(asset_id),
                category=str(category),
                cls=str(cls),
                s3_png_key=str(s3_png_key),
                distance=float(distance),
            )
        )
    return anchors


def select_anchor_pair(anchors: List[Anchor], max_second_rank: int = 6) -> List[Anchor]:
    """
    Avoid messy hybrids:

    - Always pick one primary (nearest).
    - Optionally pick a second from nearby ranks (still similar).
    """
    if not anchors:
        return []
    primary = anchors[0]
    idx = min(max_second_rank, len(anchors) - 1)
    secondary = anchors[idx] if idx > 0 else None

    out = [primary]
    if secondary and secondary.asset_id != primary.asset_id:
        out.append(secondary)
    return out


# -------------------
# LLM helpers (Responses API)
# -------------------

def llm_json(prompt: str, model: str = "gpt-4o-mini") -> Any:
    """
    Ask model to return ONLY JSON. We parse it.
    """
    full_prompt = (
        "You are a careful JSON generator.\n"
        "Return ONLY valid JSON, no explanation, no backticks.\n\n"
        + prompt
    )
    resp = client.responses.create(
        model=model,
        input=full_prompt,
    )
    text = resp.output[0].content[0].text
    return _safe_json_loads(text)


def llm_make_visual_intent(transcript: str) -> Dict[str, Any]:
    prompt = (
        "From the following brand transcript, produce a JSON object with:\n"
        "{\n"
        '  "visual_intent": string,   // describe formal design intent (geometry, symmetry, stroke weight, enclosure, negative space). No literal objects.\n'
        '  "keywords": [string, ...]  // 3-12 abstract design keywords (e.g. "woven", "radial", "monoline", "grid", "rounded"). \n'
        "}\n\n"
        f"Transcript:\n\"\"\"{transcript}\"\"\"\n"
    )
    return llm_json(prompt)


def llm_make_concept(
    transcript: str,
    logo_type: str,
    anchors: List[Anchor],
    visual_intent: str,
    keywords: List[str],
) -> Concept:
    """
    Multimodal call (text + anchor images) to design 1 concept:
    - logo_type: enforced
    - rationale: 1 sentence, <= 25 words
    - image_prompt: detailed text prompt for gpt-image-1
    """

    content: List[Dict[str, Any]] = []
    print(f"Required logo_type: {logo_type}\n\n"
                f"Visual intent: {visual_intent}\n"
                f"Keywords: {', '.join(keywords)}\n\n")

    # Text instructions + transcript
    content.append(
        {
            "type": "input_text",
            "text": (
                "You are a modernist logo designer.\n"
                "You MUST design a logo of the specified logo_type.\n"
                "Use the reference logos ONLY as loose stylistic cues.\n"
                "Do NOT create a literal hybrid collage.\n"
                "Do NOT depict the business subject literally. Use abstract/formal cues instead.\n"
                "Favor the PRIMARY reference (Image A). Image B is a minor variation.\n"
                "Focus on: geometry, symmetry, stroke weight, negative space, composition.\n"
                "Monochrome only. Strong legibility at small sizes.\n\n"
                "Definitions:\n"
                "- word_mark: primarily typographic lettering/logotype; any symbol is subordinate; stay abstract.\n"
                "- pictorial_logo: symbolic and simplified, not a literal depiction of the product/service; avoid on-the-nose objects.\n"
                "- abstract_icon: geometric, non-literal, no identifiable object.\n\n"
                f"Required logo_type: {logo_type}\n\n"
                f"Visual intent: {visual_intent}\n"
                f"Keywords: {', '.join(keywords)}\n\n"
                # f"Transcript:\n\"\"\"{transcript}\"\"\"\n"
            ),
        }
    )

    # Reference images
    for i, a in enumerate(anchors):
        label = "Image A (PRIMARY reference)" if i == 0 else "Image B (SECONDARY variation)"
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "image_url": a.data_url})

    # JSON-only instruction
    content.append(
        {
            "type": "input_text",
            "text": (
                "Now respond ONLY with JSON of the form:\n"
                "{\n"
                '  "logo_type": "word_mark" | "pictorial_logo" | "abstract_icon",  // must equal the required logo_type\n'
                '  "rationale": "one sentence, <= 25 words, explaining why this form fits the brief",\n'
                '  "image_prompt": "detailed text prompt for an AI image generator; do NOT mention the reference images; do NOT include literal depictions of the business offering."\n'
                "}\n"
            ),
        }
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content}],
    )

    raw = resp.output[0].content[0].text
    data = _safe_json_loads(raw)

    if data.get("logo_type") != logo_type:
        raise RuntimeError(f"Model violated logo_type: expected {logo_type}, got {data.get('logo_type')}")

    return Concept(
        logo_type=logo_type,
        rationale=data["rationale"].strip(),
        image_prompt=data["image_prompt"].strip(),
        anchors=anchors,
    )


def openai_generate_image(prompt: str) -> str:
    """
    Call gpt-image-1. Return b64 PNG.
    """
    res = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="high",
    )
    b64 = res.data[0].b64_json
    if not isinstance(b64, str) or not b64:
        raise RuntimeError("Image generation did not return b64_json")
    return b64


def llm_critique_and_refine(
    transcript: str,
    concept: Concept,
    image_b64: str,
) -> Tuple[str, str]:
    """
    Critique the raster logo and return (maybe_refined_prompt, final_b64).
    If the model suggests refinements, we regenerate once with an updated prompt.
    """
    img_url = _b64_to_data_url_png(image_b64)

    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are critiquing a generated logo for modernist quality.\n"
                "You must return ONLY JSON of the form:\n"
                "{\n"
                '  "scores": {\n'
                '    "clarity": 1-5,\n'
                '    "singularity": 1-5,\n'
                '    "distinctiveness": 1-5,\n'
                '    "fit": 1-5\n'
                "  },\n"
                '  "suggested_refinements": [string, ...]   // 0-3 short bullet points\n'
                "}\n"
                "Clarity: legible at small sizes.\n"
                "Singularity: one dominant form, no clutter.\n"
                "Distinctiveness: feels specific, not generic.\n"
                "Fit: matches transcript and logo_type.\n"
            ),
        },
        {"type": "input_text", "text": f"Transcript:\n{transcript}"},
        {"type": "input_text", "text": f"Logo type: {concept.logo_type}"},
        {"type": "input_text", "text": f"Generation prompt:\n{concept.image_prompt}"},
        {"type": "input_image", "image_url": img_url},
    ]

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content}],
    )
    raw = resp.output[0].content[0].text
    data = _safe_json_loads(raw)

    refinements = data.get("suggested_refinements") or []
    if not isinstance(refinements, list) or not refinements:
        # keep original prompt and image
        return concept.image_prompt, image_b64

    refinements_text = " ".join(str(r) for r in refinements[:3])
    new_prompt = concept.image_prompt + " " + refinements_text

    new_b64 = openai_generate_image(new_prompt)
    return new_prompt, new_b64


# -------------------
# Pipeline core
# -------------------

def run_pipeline(transcript: str, refine: bool = True) -> List[Concept]:
    """
    Main entry: transcript -> 3 concepts + 3 final PNGs.
    """

    # 1) Visual intent + keywords -> embed
    intent = llm_make_visual_intent(transcript)
    visual_intent = intent["visual_intent"]
    keywords = intent["keywords"]
    query_vec = embed_text_openclip(visual_intent + " | " + ", ".join(keywords))

    # 2) For each required logo type, retrieve anchors + design concept
    required_types = ["word_mark", "pictorial_logo", "abstract_icon"]
    concepts: List[Concept] = []

    for t in required_types:
        anchors = retrieve_anchors(query_vec, logo_type=t, k=25)
        anchors = select_anchor_pair(anchors, max_second_rank=6)

        # hydrate S3 -> data_url for vision step
        for a in anchors:
            a.data_url = _s3_get_png_as_data_url(LOGOBOOK_S3_BUCKET, a.s3_png_key)

        concept = llm_make_concept(
            transcript=transcript,
            logo_type=t,
            anchors=anchors,
            visual_intent=visual_intent,
            keywords=keywords,
        )
        concepts.append(concept)

    # 3) Generate + optional critique/refine per concept
    for c in concepts:
        first_b64 = openai_generate_image(c.image_prompt)
        if refine:
            final_prompt, final_b64 = llm_critique_and_refine(transcript, c, first_b64)
            c.image_prompt = final_prompt
            c.image_b64 = final_b64
        else:
            c.image_b64 = first_b64

    return concepts


# -------------------
# CLI helpers
# -------------------

def _read_transcript(path: Optional[str]) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return sys.stdin.read().strip()


def save_pngs(concepts: List[Concept], out_dir: str) -> Dict[str, Any]:
    """
    Save each concept's PNG as <logo_type>.png under out_dir,
    and return a small metadata dict you can also log/inspect.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta: Dict[str, Any] = {"concepts": []}

    for c in concepts:
        if not c.image_b64:
            continue
        filename = f"{c.logo_type}.png"
        path = os.path.join(out_dir, filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(c.image_b64))

        meta["concepts"].append(
            {
                "logo_type": c.logo_type,
                "rationale": c.rationale,
                "final_prompt": c.image_prompt,
                "file": path,
                "anchors": [
                    {
                        "asset_id": a.asset_id,
                        "category": a.category,
                        "class": a.cls,
                        "s3_png_key": a.s3_png_key,
                        "distance": a.distance,
                    }
                    for a in c.anchors
                ],
            }
        )

    return meta


# -------------------
# CLI entry
# -------------------

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, help="Path to transcript text file (or read from stdin)")
    parser.add_argument("--out-dir", type=str, help="Directory to save 3 PNGs and meta.json")
    parser.add_argument("--no-refine", action="store_true", help="Disable critique/refinement loop")
    args = parser.parse_args()

    transcript = _read_transcript(args.transcript)
    if not transcript:
        raise SystemExit("Transcript is empty.")

    concepts = run_pipeline(transcript, refine=not args.no_refine)

    if args.out_dir:
        meta = save_pngs(concepts, args.out_dir)
        meta_path = os.path.join(args.out_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Wrote 3 PNGs + metadata to: {args.out_dir}")
    else:
        # Just print JSON with base64 inline
        out = {
            "concepts": [
                {
                    "logo_type": c.logo_type,
                    "rationale": c.rationale,
                    "final_prompt": c.image_prompt,
                    "image_b64": c.image_b64,
                    "anchors": [
                        {
                            "asset_id": a.asset_id,
                            "category": a.category,
                            "class": a.cls,
                            "s3_png_key": a.s3_png_key,
                            "distance": a.distance,
                        }
                        for a in c.anchors
                    ],
                }
                for c in concepts
            ]
        }
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    cli_main()
