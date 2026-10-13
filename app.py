import os
import sqlite3
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

import recognizer

DECK_DIR = os.path.join(recognizer.YGOPRO_DIR, "deck")

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400
    f = request.files["image"]
    img = Image.open(BytesIO(f.read())).convert("RGB")

    col_hint = request.form.get("cols")
    col_hint = int(col_hint) if col_hint else None

    row_hints = {}
    for key in ("main", "extra", "side"):
        v = request.form.get(f"rows_{key}")
        if v:
            row_hints[key] = int(v)

    result = recognizer.analyze(img, col_hint=col_hint, row_hints=row_hints)
    return jsonify(result)


@app.post("/api/rebuild_cache")
def api_rebuild_cache():
    recognizer.build_hash_cache(force=True)
    recognizer._HASH_CACHE = None
    recognizer.get_hash_cache()
    return jsonify({"ok": True})


@app.post("/api/export")
def api_export():
    data = request.get_json(force=True)
    deck_name = (data.get("deck_name") or "").strip()
    if not deck_name:
        return jsonify({"error": "deck_name required"}), 400
    # keep it to characters that are safe in a Windows filename
    bad = '\\/:*?"<>|'
    if any(c in deck_name for c in bad):
        return jsonify({"error": "deck_name contains invalid filename characters"}), 400

    def expand(section):
        ids = []
        for entry in data.get(section, []):
            ids.extend([int(entry["id"])] * int(entry.get("count", 1)))
        return ids

    main = expand("main")
    extra = expand("extra")
    side = expand("side")

    lines = ["#created by Claude (deckbuilder)", "#main"]
    lines += [str(i) for i in main]
    lines.append("#extra")
    lines += [str(i) for i in extra]
    lines.append("!side")
    lines += [str(i) for i in side]

    out_path = os.path.join(DECK_DIR, f"{deck_name}.ydk")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    return jsonify({"ok": True, "path": out_path, "main": len(main), "extra": len(extra), "side": len(side)})


@app.get("/api/card/<int:card_id>")
def api_card_name(card_id):
    con = sqlite3.connect(recognizer.CDB_PATH)
    cur = con.cursor()
    cur.execute("select name from texts where id=?", (card_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": card_id, "name": row[0]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5151, debug=False)
