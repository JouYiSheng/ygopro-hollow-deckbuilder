"""
Core recognition pipeline for turning a Neuron-style deck recipe screenshot
into a list of (card_id, count) per section, using only local image
comparison against the game's own cards.cdb + pics/ folder (no OCR, no
network calls) - per the "classify by frame color, then search" approach.

Pipeline:
  1. detect_sections(img)   - find メインデッキ/エクストラデッキ/サイドデッキ
                               zones via the bright cyan header underlines.
  2. detect_grid(img, zone) - find the card grid (columns via autocorrelation,
                               rows via the calibrated card aspect ratio).
  3. classify_frame(cell)   - sample the name-bar color to guess a coarse
                               type bucket (spell / trap / effect / ... ).
  4. match_card(cell, bucket) - average-hash the art region against a cached,
                               type-filtered pool of pics/*.jpg and return the
                               top-N closest matches.

Everything here is pure, local, offline computation - no card name text is
ever read.
"""
import os
import pickle
import sqlite3
import base64
from io import BytesIO

import numpy as np
from PIL import Image

# Root of the local simulator install - each machine may have it on a
# different drive/path, so this is overridable via env var instead of
# hardcoded, letting the same checkout run on multiple computers.
YGOPRO_DIR = os.environ.get("YGOPRO_DIR", r"H:\YGOPro\ygopro ver.1.035.4 hollow")
CDB_PATH = os.path.join(YGOPRO_DIR, "cards.cdb")
PICS_DIR = os.path.join(YGOPRO_DIR, "pics")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
HASH_CACHE_PATH = os.path.join(CACHE_DIR, "hash_cache.pkl")

# Art-crop ratios (relative to one card cell), calibrated against both the
# Neuron deck-recipe template and the official pics/ card renders.
ART_X0, ART_Y0, ART_X1, ART_Y1 = 0.08, 0.11, 0.92, 0.62

# Name-bar sampling ratios, used only to classify the frame color.
BAR_X0, BAR_Y0, BAR_X1, BAR_Y1 = 0.08, 0.03, 0.75, 0.10

# zone_row_height ≈ column_width * ROW_ASPECT (calibrated from two real
# Neuron exports; both main/extra/side rows follow the same card aspect
# ratio in that template).
ROW_ASPECT = 1.43

# Rows/columns below this stdev are treated as an empty placeholder cell.
EMPTY_STD_THRESHOLD = 14.0

# If every cell in a detected row has a best-candidate hash distance above
# this, the row is almost certainly footer text/logos past the real grid
# rather than actual cards - stop there instead of trusting the zone height.
NOISE_ROW_DIST_THRESHOLD = 35

FRAME_BUCKETS = {
    # bucket name -> (reference RGB, SQL WHERE fragment on datas.type)
    # reference RGB calibrated with sample_frame_color() (see below), not a
    # raw pixel pick - must stay in sync if that sampling method changes.
    "normal":  ((184, 147, 84),  "type&1!=0 and type&0x20=0 and type&0x40=0 and type&0x2000=0 and type&0x800000=0 and type&0x4000000=0 and type&0x80=0"),
    "effect":  ((172, 98, 57),   "type&0x20!=0 and type&0x40=0 and type&0x80=0 and type&0x2000=0 and type&0x800000=0 and type&0x4000000=0 and type&0x1000000=0"),
    "ritual":  ((92, 123, 175),  "type&0x80!=0 and type&1!=0"),
    "fusion":  ((130, 65, 145),  "type&0x40!=0"),
    "synchro": ((157, 154, 152), "type&0x2000!=0"),
    "xyz":     ((93, 94, 94),    "type&0x800000!=0"),
    "link":    ((48, 108, 158),  "type&0x4000000!=0"),
    "spell":   ((37, 157, 140),  "type&2!=0"),
    "trap":    ((176, 60, 132),  "type&4!=0"),
}
FALLBACK_BUCKET_SQL = "type&1!=0 or type&2!=0 or type&4!=0"  # everything


def get_conn():
    return sqlite3.connect(CDB_PATH)


# ---------------------------------------------------------------- hashing --

def ahash(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((16, 16))
    arr = np.array(g, dtype=np.float32)
    return arr > arr.mean()


def art_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * ART_X0), int(h * ART_Y0), int(w * ART_X1), int(h * ART_Y1)))


def build_hash_cache(force: bool = False) -> dict:
    """id -> 16x16 bool ndarray, for every pics/*.jpg. Cached to disk."""
    if not force and os.path.exists(HASH_CACHE_PATH):
        with open(HASH_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    for name in os.listdir(PICS_DIR):
        if not name.lower().endswith(".jpg"):
            continue
        stem = name[:-4]
        if not stem.isdigit():
            continue
        cid = int(stem)
        try:
            img = Image.open(os.path.join(PICS_DIR, name))
            cache[cid] = ahash(art_crop(img))
        except Exception:
            continue
    with open(HASH_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    return cache


_HASH_CACHE = None


def get_hash_cache():
    global _HASH_CACHE
    if _HASH_CACHE is None:
        _HASH_CACHE = build_hash_cache()
    return _HASH_CACHE


# --------------------------------------------------------- section finder --

def _is_cyan(px):
    r, g, b = px
    return 130 <= r <= 235 and 200 <= g <= 255 and 230 <= b <= 255 and g >= r and b >= g


def detect_sections(img: Image.Image):
    """Return [(label, y_start, y_end), ...] for main/extra/side zones,
    using the bright cyan header-bar borders as landmarks."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    px = rgb.load()
    step = max(1, w // 60)

    cyan_rows = []
    for y in range(h):
        hits = 0
        total = 0
        for x in range(0, w, step):
            total += 1
            if _is_cyan(px[x, y]):
                hits += 1
        if total and hits / total > 0.6:
            cyan_rows.append(y)

    # group consecutive cyan rows into bands
    bands = []
    for y in cyan_rows:
        if bands and y - bands[-1][-1] <= 3:
            bands[-1].append(y)
        else:
            bands.append([y])
    bands = [(b[0], b[-1]) for b in bands]

    # bands alternate: header-top, header-bottom, header-top, header-bottom, ...
    zones = []
    labels = ["main", "extra", "side"]
    for i in range(1, len(bands), 2):
        zone_start = bands[i][1] + 1
        zone_end = bands[i + 1][0] - 1 if i + 1 < len(bands) else h
        label = labels[len(zones)] if len(zones) < len(labels) else f"section{len(zones)+1}"
        zones.append((label, zone_start, zone_end))
    return zones


# ------------------------------------------------------------ grid finder --

def _autocorr_period(profile: np.ndarray, lo: int, hi: int) -> int:
    prof = profile - profile.mean()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    lo = max(lo, 1)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return max(1, len(profile))
    return lo + int(np.argmax(ac[lo:hi]))


def detect_columns(img: Image.Image, y0: int, y1: int, col_hint: int = None) -> int:
    if col_hint:
        return col_hint
    gray = np.array(img.convert("L"))[y0:y1, :]
    w = gray.shape[1]
    colprof = gray.mean(axis=0)
    period = _autocorr_period(colprof, w // 15, w // 3)
    cols = max(1, round(w / period))
    return cols


def cell_is_empty(cell: Image.Image) -> bool:
    arr = np.array(cell.convert("L"), dtype=np.float32)
    return float(arr.std()) < EMPTY_STD_THRESHOLD


def sample_frame_color(img: Image.Image, box) -> tuple:
    """Robust background-color estimate for a name-bar strip: drop
    near-white/near-black pixels (text, outlines, icons) and average the
    rest. Far more stable than a raw color mode, which is easily thrown off
    by anti-aliasing/JPEG noise when the strip is only a few pixels tall."""
    region = np.array(img.crop(box).convert("RGB"), dtype=np.float32).reshape(-1, 3)
    brightness = region.sum(axis=1)
    mask = (brightness > 60) & (brightness < 650)
    filtered = region[mask] if mask.sum() > 5 else region
    return tuple(filtered.mean(axis=0))


def classify_frame(cell: Image.Image, top_k: int = 2) -> list:
    """Return the top_k closest frame-color buckets, best first. The frame
    color alone can't always tell apart neighbouring buckets (e.g. a dark
    Link frame vs a dark-ish Ritual blue under JPEG noise), so callers merge
    candidate pools across the top few buckets instead of trusting just one."""
    w, h = cell.size
    box = (int(w * BAR_X0), int(h * BAR_Y0), int(w * BAR_X1), int(h * BAR_Y1))
    c = np.array(sample_frame_color(cell, box), dtype=np.float32)
    dists = []
    for name, (ref, _sql) in FRAME_BUCKETS.items():
        d = float(np.linalg.norm(c - np.array(ref, dtype=np.float32)))
        dists.append((d, name))
    dists.sort()
    return [name for _d, name in dists[:top_k]]


def candidate_ids_for_buckets(cur, buckets):
    ids = set()
    for bucket in buckets:
        sql = FRAME_BUCKETS.get(bucket, (None, FALLBACK_BUCKET_SQL))[1]
        cur.execute(f"select id from datas where {sql}")
        ids.update(r[0] for r in cur.fetchall())
    return ids


def match_card(cell: Image.Image, buckets, cur, top_n: int = 3):
    cache = get_hash_cache()
    ids = candidate_ids_for_buckets(cur, buckets)
    qh = ahash(art_crop(cell))
    results = []
    for cid in ids:
        h = cache.get(cid)
        if h is None:
            continue
        dist = int(np.count_nonzero(qh != h))
        results.append((dist, cid))
    results.sort()
    top = results[:top_n]
    out = []
    for dist, cid in top:
        cur.execute("select name from texts where id=?", (cid,))
        row = cur.fetchone()
        out.append({"id": cid, "name": row[0] if row else "?", "dist": dist})
    return out


def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def analyze(image: Image.Image, col_hint: int = None, row_hints: dict = None):
    """Full pipeline. Returns {sections: [{label, cells:[...]}]}"""
    row_hints = row_hints or {}
    zones = detect_sections(image)
    con = get_conn()
    cur = con.cursor()

    out_sections = []
    for label, y0, y1 in zones:
        cols = detect_columns(image, y0, y1, col_hint)
        col_w = image.size[0] / cols
        zone_h = y1 - y0
        row_hint = row_hints.get(label)
        row_h = zone_h / row_hint if row_hint else col_w * ROW_ASPECT

        cells = []
        r = 0
        max_rows = row_hint if row_hint else 25
        while r < max_rows:
            cy0 = int(y0 + r * row_h)
            cy1 = int(y0 + (r + 1) * row_h)
            if cy0 >= y1:
                break
            cy1 = min(cy1, y1)

            row_cells = []
            for c in range(cols):
                cx0 = int(c * col_w)
                cx1 = int((c + 1) * col_w)
                cell = image.crop((cx0, cy0, cx1, cy1))
                if cell_is_empty(cell):
                    continue
                buckets = classify_frame(cell)
                candidates = match_card(cell, buckets, cur, top_n=3)
                row_cells.append({
                    "row": r, "col": c,
                    "bucket": "/".join(buckets),
                    "thumb": img_to_b64(cell),
                    "candidates": candidates,
                })

            if not row_hint:
                best_dists = [c["candidates"][0]["dist"] for c in row_cells if c["candidates"]]
                if not row_cells or (r > 0 and best_dists and min(best_dists) >= NOISE_ROW_DIST_THRESHOLD):
                    # trailing empty row, or a row that's actually footer
                    # text/logos past the real grid (every "match" in it is
                    # a bad one) - stop instead of trusting the zone height.
                    break

            cells.extend(row_cells)
            r += 1

        out_sections.append({
            "label": label, "y0": y0, "y1": y1,
            "cols": cols, "rows": r,
            "cells": cells,
        })

    con.close()
    return {"sections": out_sections}
