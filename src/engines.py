"""The OCR engines, behind one interface.

Every engine returns the same thing — a list of Word boxes — so the field
extractor can work on geometry (which column is this number in?) rather than
on whatever each engine does to line breaks. A bill's money column is the part
that matters and the part plain text loses: "250.00 1.000 59.99" tells you
nothing about which figure is the rate and which the amount, while the x
positions do.

Engines are listed in `ENGINES` with their cost, so the bench can report
accuracy against seconds rather than accuracy alone. EasyOCR is imported
lazily: loading torch costs several seconds even when it is never used.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

DIGITS_WHITELIST = "0123456789.,-+₹RsCrDr/%"


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> int:
        return self.x + self.w


def _encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode the page for OCR")
    return buf.tobytes()


def _run_tesseract(image: np.ndarray, args: list[str]) -> str:
    try:
        done = subprocess.run(["tesseract", "stdin", "stdout", *args],
                              input=_encode(image), capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return ""
    return done.stdout.decode("utf-8", "replace")


def tesseract_text(image: np.ndarray, psm: int = 6) -> str:
    return _run_tesseract(image, ["--psm", str(psm)])


def tesseract_words(image: np.ndarray, psm: int = 6, whitelist: str | None = None) -> list[Word]:
    """Word boxes via TSV. This is the only tesseract call that gives
    geometry, and geometry is what the column reader needs."""
    args = ["--psm", str(psm), "tsv"]
    if whitelist:
        args = ["--psm", str(psm), "-c", f"tessedit_char_whitelist={whitelist}", "tsv"]
    out = _run_tesseract(image, args)
    words: list[Word] = []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        text = parts[11].strip()
        if not text:
            continue
        try:
            conf = float(parts[10])
            x, y, w, h = (int(parts[i]) for i in (6, 7, 8, 9))
        except ValueError:
            continue
        if conf < 0:
            continue
        words.append(Word(text=text, x=x, y=y, w=w, h=h, conf=conf / 100.0))
    return words


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr  # imported here: loading torch costs seconds even unused
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def easyocr_words(image: np.ndarray) -> list[Word]:
    results = _easyocr_reader().readtext(image, detail=1, paragraph=False)
    words: list[Word] = []
    for box, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        words.append(Word(text=text, x=min(xs), y=min(ys),
                          w=max(xs) - min(xs), h=max(ys) - min(ys), conf=float(conf)))
    return words


@lru_cache(maxsize=1)
def _rapidocr_engine():
    # Lazily imported for the same reason easyocr is: loading onnxruntime
    # costs real time even when this engine is never run. RapidOCR ships
    # the PP-OCRv4 detection+recognition models (the same lineage a full
    # PaddleOCR install would use) over onnxruntime, with no CUDA/paddle
    # framework dependency — a suggested candidate worth measuring, not
    # trusted on reputation, same as every other method here.
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def rapidocr_words(image: np.ndarray) -> list[Word]:
    result, _ = _rapidocr_engine()(image)
    words: list[Word] = []
    for box, text, conf in (result or []):
        text = str(text).strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append(Word(text=text, x=int(min(xs)), y=int(min(ys)),
                          w=int(max(xs) - min(xs)), h=int(max(ys) - min(ys)), conf=float(conf)))
    return words


def words_to_lines(words: list[Word], tolerance: float = 0.6) -> list[list[Word]]:
    """Group words into printed lines by vertical overlap.

    Row height drives the tolerance rather than a fixed pixel count, so it
    behaves the same on a 900 px screenshot and a 2400 px photo.
    """
    if not words:
        return []
    heights = [w.h for w in words if w.h > 0]
    band = (float(np.median(heights)) if heights else 10.0) * tolerance
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.cy, w.x)):
        for line in lines:
            if abs(line[0].cy - word.cy) <= band:
                line.append(word)
                break
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.x) for line in lines]


def line_text(line: list[Word]) -> str:
    return " ".join(w.text for w in sorted(line, key=lambda w: w.x)).strip()


def text_from_words(words: list[Word]) -> str:
    return "\n".join(line_text(line) for line in words_to_lines(words))


# name -> (callable(image) -> list[Word], rough seconds per page)
ENGINES: dict[str, tuple] = {
    "tess-psm6": (lambda img: tesseract_words(img, psm=6), 1.5),
    "tess-psm4": (lambda img: tesseract_words(img, psm=4), 1.5),
    "tess-psm11": (lambda img: tesseract_words(img, psm=11), 2.0),
    "easyocr": (easyocr_words, 12.0),
    "rapidocr": (rapidocr_words, 2.0),
}


def run_engine(name: str, image: np.ndarray) -> list[Word]:
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; have {', '.join(ENGINES)}")
    return ENGINES[name][0](image)


_NUMERIC = re.compile(r"^[₹Rs.]*[\d,]+\.?\d*$", re.I)


def reread_numbers(image: np.ndarray, line: list[Word]) -> list[Word]:
    """Re-read one line's numeric tail with a digit whitelist.

    The money column is where every engine fails first (a real bill's 460.82
    came back as "480 82"), and constraining the alphabet is the cheapest fix
    that does not need another model.
    """
    numeric = [w for w in line if _NUMERIC.match(w.text)]
    if not numeric:
        return []
    pad = 6
    x0 = max(min(w.x for w in numeric) - pad, 0)
    y0 = max(min(w.y for w in numeric) - pad, 0)
    x1 = min(max(w.right for w in numeric) + pad, image.shape[1])
    y1 = min(max(w.y + w.h for w in numeric) + pad, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return []
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    words = tesseract_words(crop, psm=7, whitelist=DIGITS_WHITELIST)
    return [Word(text=w.text, x=x0 + w.x // 2, y=y0 + w.y // 2,
                 w=max(w.w // 2, 1), h=max(w.h // 2, 1), conf=w.conf) for w in words]
