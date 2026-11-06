"""Importing photos into the widget's database, and filing the obvious ones.

The image is copied into the widget's `data/` tree rather than referenced
where it sits: the phone drops bills on a removable drive, in a folder the
sync app prunes, and a thumbnail that breaks when the drive is unplugged is
worse than a copy.

Identity is the sha256 of the image bytes, so importing the same folder twice
costs nothing and a renamed copy collapses onto the original.

Auto-filing is deliberately timid. It only ever links to an **untracked**
transaction, needs a high score and a clear lead over the runner-up, and
leaves everything else for a human. A wrong link is worse than no link.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from . import match as M
from . import store
from .parse import parse_file
from .validate import Parse

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_WIDTH = 600


@dataclass
class Imported:
    receipt_id: str
    path: Path
    parse: Parse
    linked_to: str | None = None
    score: int | None = None
    reasons: list[str] | None = None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store_root() -> Path:
    return store.DB_PATH.parent / "receipts"


def store_image(path: Path, sha: str) -> tuple[str, str | None]:
    """Copy the photo in and make a thumbnail. Returns paths relative to
    `data/`, which is what the database holds."""
    root = _store_root()
    folder = root / sha[:2]
    old = os.umask(0o077)                      # these are personal purchases
    try:
        folder.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        target = folder / f"{sha}{path.suffix.lower()}"
        if not target.exists():
            shutil.copy2(path, target)
            target.chmod(0o600)
        thumb_dir = root / "thumb"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb = thumb_dir / f"{sha}.jpg"
        if not thumb.exists():
            image = cv2.imread(str(target))
            if image is not None:
                h, w = image.shape[:2]
                scale = min(1.0, THUMB_WIDTH / max(w, 1))
                small = cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)),
                                   interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(thumb), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                thumb.chmod(0o600)
    finally:
        os.umask(old)
    return (str(target.relative_to(store.DB_PATH.parent)),
            str(thumb.relative_to(store.DB_PATH.parent)) if thumb.exists() else None)


def to_row(parse: Parse, path: Path, sha: str, stored: str, thumb: str | None) -> dict:
    """The shape txdb.receipts.save_parse takes."""
    stat = path.stat()
    return {
        "image_sha256": sha, "stored_path": stored, "thumb_path": thumb,
        "source_path": str(path),
        "source_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "source_size": stat.st_size,
        "rotation": parse.rotation, "merchant": parse.merchant, "gstin": parse.gstin,
        "date": parse.date, "time": parse.time, "invoice_no": parse.invoice_no,
        "total_paise": _paise(parse.total), "subtotal_paise": _paise(parse.subtotal),
        "tax_paise": _paise(parse.tax_total), "discount_paise": _paise(parse.discount),
        "round_off_paise": _paise(parse.round_off),
        "payment_hint": parse.payment_hint, "before_payment": parse.is_cart,
        "method": parse.method, "confidence": parse.confidence,
        "needs_review": parse.needs_review, "warnings": parse.warnings,
        "raw_text": parse.raw_text, "parser_version": "1",
        "items": [{"description": i["description"], "amount": i["amount"],
                  "qty": i.get("qty"), "unit": i.get("unit"),
                  "rate": i.get("rate"), "hsn": i.get("hsn")} for i in parse.items],
        "taxes": _taxes(parse),
    }


def _paise(value) -> int | None:
    return None if value is None else int(round(float(value) * 100))


def _taxes(parse: Parse) -> list[dict]:
    if not any((parse.cgst, parse.sgst, parse.igst, parse.cess, parse.tax_total)):
        return []
    return [{"rate": 0.0, "taxable": parse.subtotal, "cgst": parse.cgst, "sgst": parse.sgst,
             "igst": parse.igst, "cess": parse.cess}]


def reparse_receipt(receipt_id: str, method: str = "H-vision-llm-gemma", on_chunk=None) -> Parse:
    """Re-read one already-imported receipt's own stored photo.

    Looks up the original source photo this receipt was imported from and
    runs `parse_file` on it again, then writes the result back through the
    same idempotent `save_parse` the bulk `--reparse` flag uses: items and
    taxes are replaced, but the row's identity (its sha256) is unchanged, so
    `receipt_links` — the user's own filing decision — is never touched.
    """
    import txdb.receipts as txdb_receipts
    from txdb import db as txdb_db

    conn = txdb_db.connect()
    row = conn.execute(
        "SELECT source_path, image_sha256, stored_path, thumb_path "
        "FROM receipts WHERE id = ?", (receipt_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no receipt with id {receipt_id!r}")
    source_path, sha, stored, thumb = row
    photo = Path(source_path)
    if not photo.exists():
        raise SystemExit(f"the original photo is gone: {source_path}")
    parse, _ = parse_file(photo, method, on_chunk=on_chunk)
    txdb_receipts.save_parse(conn, to_row(parse, photo, sha, stored, thumb))
    return parse


def photos_in(folder: Path) -> list[Path]:
    if not folder.exists():
        raise SystemExit(f"{folder} is not there (is the drive mounted?)")
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES),
                  key=lambda p: p.stat().st_mtime)


def scan(folder: Path, method: str = "H-vision-llm-gemma", apply: bool = True,
         reparse: bool = False, on_progress=None, on_start=None, on_chunk=None) -> list[Imported]:
    """Read every photo in `folder` and file the obvious ones.

    With `apply=False` nothing is written: the same work, printed instead of
    saved, which is how a run is checked before it touches the database.

    `on_start(photo)`, when given, fires right before that photo's own read
    begins - the only way a caller can print "now reading X" rather than
    going quiet for the whole scan and reporting everything at the end.
    `on_chunk` is the same per-token callback `parse_file` already takes,
    passed straight through for an LLM method.
    """
    import txdb.receipts as txdb_receipts           # noqa: E402  (needs TXDB_HOME)
    from txdb import db as txdb_db

    conn = txdb_db.connect() if apply else None
    known = set()
    if conn is not None and not reparse:
        known = {r[0] for r in conn.execute("SELECT image_sha256 FROM receipts")}

    ledger = store.transactions(store.connect())
    taken = {r[0] for r in conn.execute(
        "SELECT transaction_id FROM receipt_links WHERE transaction_id IS NOT NULL")} \
        if conn is not None else set()

    out: list[Imported] = []
    for photo in photos_in(folder):
        sha = sha256_of(photo)
        if sha in known:
            continue
        if on_start:
            on_start(photo)
        parse, _ = parse_file(photo, method, on_chunk=on_chunk)
        if on_progress:
            on_progress(photo, parse)

        entry = Imported(receipt_id=txdb_receipts.receipt_id(sha), path=photo, parse=parse)
        candidates = [c for c in M.rank({"merchant": parse.merchant, "date": parse.date,
                                         "total": parse.total},
                                        ledger, untracked_only=True, limit=5)
                      if c.transaction["id"] not in taken]
        picked = M.auto_pick(candidates)
        if picked:
            entry.linked_to = picked.transaction["id"]
            entry.score = picked.score
            entry.reasons = picked.reasons

        if apply:
            stored, thumb = store_image(photo, sha)
            rid = txdb_receipts.save_parse(conn, to_row(parse, photo, sha, stored, thumb))
            entry.receipt_id = rid
            if picked:
                txdb_receipts.set_link(conn, rid, transaction_id=picked.transaction["id"],
                                       matched_by="auto", score=picked.score)
                taken.add(picked.transaction["id"])
        out.append(entry)
    return out
