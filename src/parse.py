"""The methods: each one a recipe of (engine, rendering) pairs, all feeding
the same referee.

They exist so the bench can answer a question with measurements instead of
opinion — is a slow deep model worth its seconds on these bills, or does
cheap OCR plus arithmetic get there? Adding a method is adding one entry to
METHODS; nothing else changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import fields as F
from . import llm
from .engines import run_engine, text_from_words
from .normalize import Page, prepare
from .validate import Parse, add_warnings, referee, score_confidence

# method name -> [(engine, rendering), ...]
METHODS: dict[str, list[tuple[str, str]]] = {
    "A-tess6":      [("tess-psm6", "gray")],
    "B-tess4":      [("tess-psm4", "gray")],
    "C-tess6-bin":  [("tess-psm6", "sauvola"), ("tess-psm6", "otsu")],
    "D-easyocr":    [("easyocr", "gray")],
    "E-consensus":  [("tess-psm6", "gray"), ("tess-psm4", "gray"), ("tess-psm6", "sauvola"),
                     ("tess-psm6", "otsu"), ("tess-psm6", "flat"), ("easyocr", "gray")],
    # A friend's suggested stack used full PaddleOCR; this is the same
    # PP-OCRv4 model lineage without the heavy paddle framework or a GPU
    # dependency (rapidocr-onnxruntime). Measured here, not assumed —
    # a dedicated document/receipt OCR engine SHOULD beat general-purpose
    # tesseract/EasyOCR on this kind of layout, but "should" is a
    # hypothesis for bench to confirm or reject like every other method.
    "J-rapidocr":   [("rapidocr", "gray")],
    "K-consensus+rapid": [("tess-psm6", "gray"), ("tess-psm4", "gray"), ("tess-psm6", "sauvola"),
                          ("tess-psm6", "otsu"), ("tess-psm6", "flat"), ("easyocr", "gray"),
                          ("rapidocr", "gray")],
}

# The plan's Stage C bake-off: a local LLM, run through ollama on this
# machine only (see llm.py — the loopback address is not configurable).
# method name -> ("text" | "vision", ollama model name). A text method
# reconciles already-OCR'd text; a vision method reads the photo directly.
# Neither is a default: they earn a place in METHODS only if `bench` shows
# they are worth their seconds over E-consensus.
LLM_METHODS: dict[str, tuple[str, str]] = {
    "F-text-llm": ("text", llm.TEXT_MODEL),
    "G-vision-llm": ("vision", llm.VISION_MODEL),
    "G2-vision-llm-alt": ("vision", llm.VISION_MODEL_ALT),
    "H-vision-llm-gemma": ("vision", llm.VISION_MODEL_GEMMA),
}
ALL_METHODS: list[str] = [*METHODS, *LLM_METHODS]


@dataclass
class Reading:
    """What one (engine, rendering) pair saw."""
    engine: str
    variant: str
    doc: F.Doc
    confidence: float = 0.0
    seconds: float = 0.0
    candidates: dict = field(default_factory=dict)
    items_by_layout: dict = field(default_factory=dict)


def exif_date(path: Path) -> str | None:
    """When the photo was taken — the fallback for bills that print no date,
    and the only date a cart screenshot has."""
    try:
        with Image.open(path) as im:
            taken = (im.getexif() or {}).get(36867) or (im.getexif() or {}).get(306)
    except Exception:
        return None
    if not taken:
        return None
    try:
        return str(taken)[:10].replace(":", "-")
    except Exception:
        return None


def reconcile_date(read_date: str | None, shot_date: str | None,
                   warnings: list[str]) -> str | None:
    """Settle the bill's printed date against the photo's timestamp.

    A real bill dated 05/11/2026 was read as 2025: a single digit, and a
    whole year wrong, which would hide the matching transaction completely.
    The photo's own timestamp is the cross-check. The day and month come
    from the bill (the photo may be taken later), the year from the camera
    when the two disagree but the day and month line up.
    """
    if not read_date:
        if shot_date:
            warnings.append("no date on the bill; using the photo's timestamp")
        return shot_date
    if not shot_date:
        return read_date
    if read_date == shot_date:
        return read_date
    if read_date[5:] == shot_date[5:] and read_date[:4] != shot_date[:4]:
        warnings.append(f"bill reads {read_date}, photo taken {shot_date}; "
                        "took the year from the photo")
        return shot_date
    return read_date


def extract(doc: F.Doc) -> tuple[dict, dict]:
    """Candidate values per field, plus the item rows each layout found."""
    lines = doc.line_texts
    candidates = {
        "total": F.find_total(lines),
        "subtotal": F.find_labelled_value(lines, F.SUBTOTAL_LABELS),
        "discount": F.find_labelled_value(lines, F.DISCOUNT_LABELS),
        "round_off": F.find_labelled_value(lines, F.ROUNDOFF_LABELS),
        "tax_total": F.find_labelled_value(lines, F.TAX_LABELS),
        "cgst": F.find_labelled_value(lines, F.CGST_LABELS),
        "sgst": F.find_labelled_value(lines, F.SGST_LABELS),
        "igst": F.find_labelled_value(lines, F.IGST_LABELS),
        "cess": F.find_labelled_value(lines, F.CESS_LABELS),
    }
    items = {"single": F.items_single_line(lines), "two-line": F.items_two_line(lines)}
    return candidates, items


def read(page: Page, engine: str, variant: str) -> Reading:
    image = next((v.image for v in page.variants if v.name == variant), page.base)
    started = time.monotonic()
    words = run_engine(engine, image)
    seconds = time.monotonic() - started
    doc = F.Doc.build(words)
    confidence = (sum(w.conf for w in words) / len(words)) if words else 0.0
    candidates, items = extract(doc)
    return Reading(engine=engine, variant=variant, doc=doc, confidence=confidence,
                   seconds=seconds, candidates=candidates, items_by_layout=items)


def _anchor_in_text(text: str) -> float | None:
    """The largest amount printed in full (with a rupee mark or decimals)
    anywhere in `text`. It is the only defence against a scale error, which
    the bill's own arithmetic cannot see."""
    best = 0.0
    for match in F.MONEY_RE.finditer(text):
        value = F.money_value(match.group(0))
        if value and value > best:
            best = value
    return best or None


def _printed_anchor(readings: list[Reading]) -> float | None:
    return _anchor_in_text("\n".join(r.doc.text for r in readings))


def _merge(readings: list[Reading]) -> dict:
    """Pool the candidate values, keeping each reading's own order.

    Order is the prior: a figure the first engine read confidently leads, and
    the referee only departs from it when the arithmetic says otherwise.
    """
    merged: dict[str, list[float]] = {}
    for reading in readings:
        for name, values in reading.candidates.items():
            bucket = merged.setdefault(name, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return merged


def _merge_items(readings: list[Reading]) -> dict:
    """Every reading's item rows, kept apart by layout and by reading.

    The temptation is to keep whichever reading found the most rows, but row
    count is not the same as row *correctness*: a noisier rendering can split
    one printed line into two and out-count a clean reading that got every
    row right. So nothing is merged or picked here - every candidate row-set
    survives, and validate.referee() is the one place that judges them, the
    same way it already judges candidate totals: by which one actually sums
    to the bill's own figures.
    """
    out: dict[str, list[list[dict]]] = {}
    for reading in readings:
        for layout, rows in reading.items_by_layout.items():
            if rows:
                out.setdefault(layout, []).append(rows)
    return out


def parse_page(page: Page, method: str) -> tuple[Parse, float]:
    """Run one method over an already-prepared page."""
    specs = METHODS[method]
    readings = [read(page, engine, variant) for engine, variant in specs]
    readings = [r for r in readings if r.doc.words]
    if not readings:
        parse = Parse(method=method, warnings=["no text could be read from this photo"])
        return parse, sum(r.seconds for r in readings)

    best_text = max(readings, key=lambda r: len(r.doc.text))
    lines = best_text.doc.line_texts
    whole = "\n".join(r.doc.text for r in readings)

    parse = referee(_merge(readings), _merge_items(readings), anchor=_printed_anchor(readings))
    parse.method = method
    parse.rotation = page.rotation
    parse.raw_text = best_text.doc.text
    parse.merchant = F.find_merchant(lines)
    parse.date = F.find_date(lines)
    parse.time = F.find_time(best_text.doc.text)
    parse.gstin = F.find_gstin(whole)
    parse.invoice_no = F.find_invoice_no(lines)
    parse.payment_hint = F.find_payment_hint(whole)
    parse.is_cart = F.looks_like_cart(whole)
    parse.confidence = score_confidence(parse, max(r.confidence for r in readings))
    add_warnings(parse)
    return parse, sum(r.seconds for r in readings)


def _llm_candidates(data: dict) -> tuple[dict, dict]:
    """The model's JSON reply, in the same shapes referee() already judges:
    a list of candidate values per field, and item rows keyed by a layout
    name. There is only ever one candidate per field here — the model gave
    one answer — but going through the same referee still earns the LLM
    reading the same arithmetic scoring and the same scale-error guard every
    other method gets, rather than a separate, uncomparable code path.
    """
    def one(value):
        return [value] if isinstance(value, (int, float)) else []

    candidates = {"total": one(data.get("total")), "subtotal": one(data.get("subtotal")),
                 "discount": one(data.get("discount")), "round_off": one(data.get("round_off"))}
    taxes = [t for t in (data.get("taxes") or []) if isinstance(t, dict)]
    if taxes:
        first = taxes[0]
        for key in ("cgst", "sgst", "igst", "cess"):
            if isinstance(first.get(key), (int, float)):
                candidates[key] = [first[key]]
        if not candidates["subtotal"] and isinstance(first.get("taxable"), (int, float)):
            candidates["subtotal"] = [first["taxable"]]

    def num(value):
        return value if isinstance(value, (int, float)) else None

    def text(value):
        return str(value).strip() or None if isinstance(value, str) else None

    rows = [{"description": str(item.get("description") or "").strip() or "(item)",
            "amount_candidates": one(item.get("amount")),
            "qty": num(item.get("qty")), "unit": text(item.get("unit")),
            "rate": num(item.get("rate")), "hsn": text(item.get("hsn"))}
           for item in (data.get("items") or []) if isinstance(item, dict)]
    rows = [r for r in rows if r["amount_candidates"]]
    return candidates, ({"llm": [rows]} if rows else {})


def parse_page_llm(page: Page, kind: str, model: str, quick_text: str = "",
                   on_chunk=None) -> tuple[Parse, float]:
    """Run one LLM method over an already-prepared page.

    `quick_text` is a fast tesseract pass taken regardless of `kind`: a text
    method reconciles it, and a vision method still gets it as the anchor
    against a scale error and as a fallback if the model is unreachable.

    `on_chunk`, when given, is called with each text fragment as the model
    streams its reply - a caller showing a person "the model is working"
    rather than a silent wait of a minute or more.
    """
    started = time.monotonic()
    if kind == "vision":
        data = (llm.structure_from_image(page.color, model=model, on_chunk=on_chunk)
               if page.color is not None else {})
    else:
        data = llm.structure_from_text(quick_text, model=model, on_chunk=on_chunk)
    seconds = time.monotonic() - started

    candidates, items_by_layout = _llm_candidates(data)
    parse = referee(candidates, items_by_layout, anchor=_anchor_in_text(quick_text))
    parse.method = f"{kind}-llm:{model}"
    parse.rotation = page.rotation
    parse.raw_text = quick_text
    parse.merchant = (data.get("merchant") or "").strip().upper() or None
    parse.date = F.parse_date(data.get("date")) if data.get("date") else None
    parse.time = data.get("time") if isinstance(data.get("time"), str) else None
    parse.gstin = data.get("gstin")
    parse.invoice_no = data.get("invoice_no")
    parse.payment_hint = (data.get("payment_hint") or None)
    parse.is_cart = bool(data.get("is_cart"))
    parse.confidence = score_confidence(parse, 1.0 if data else 0.0)
    if not data:
        parse.warnings.append(
            "the local LLM returned nothing - is ollama running, and is the model pulled "
           f"('ollama pull {model}')?")
    add_warnings(parse)
    return parse, seconds


def parse_file(path: str | Path, method: str = "E-consensus", on_chunk=None) -> tuple[Parse, float]:
    """Photo in, one reading out. The photo's timestamp fills in a date the
    bill never printed.

    `on_chunk` only has an effect for an LLM method; it is ignored by the
    deterministic OCR methods, which have no model reply to stream.
    """
    path = Path(path)
    started = time.monotonic()
    page = prepare(path)
    if method in LLM_METHODS:
        kind, model = LLM_METHODS[method]
        quick_text = text_of(page, "tess-psm6", "gray")
        parse, _ = parse_page_llm(page, kind, model, quick_text=quick_text, on_chunk=on_chunk)
    else:
        parse, _ = parse_page(page, method)
    parse.date = reconcile_date(parse.date, exif_date(path), parse.warnings)
    return parse, time.monotonic() - started


def text_of(page: Page, engine: str = "tess-psm6", variant: str = "gray") -> str:
    return text_from_words(run_engine(engine, next(
        (v.image for v in page.variants if v.name == variant), page.base)))
