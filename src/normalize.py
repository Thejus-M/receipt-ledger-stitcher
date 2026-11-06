"""Getting a photographed bill into a shape OCR can read.

Two of the ten real bills are unreadable without this: one was photographed
sideways and one upside down, and plain OCR returns gibberish for both. The
rest are thermal paper shot under room light, where the win comes from
flattening the lighting and thresholding.

The module deliberately produces *several* renderings of the same page rather
than one "best" one. Measured on a real bill, the raw grayscale reads its
discount as 481 and a sharpened+thresholded copy reads the same line as 461
(the truth is 4.61): different renderings win on different fields, and
validate.py decides between them afterwards using the bill's own arithmetic.
So this module's job is variety, not perfection.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Enough resolution for OCR without making every engine crawl: thermal print
# needs roughly this to keep the decimal points.
TARGET_WIDTH = 1600
MAX_PIXELS = 4_000_000
# OSD needs the letters big enough to recognise. At 1000 px wide a sideways
# bill came back as "90 degrees, confidence 0.14" - noise - and the page was
# then rotated the wrong way and lost entirely.
OSD_WIDTH = 2200
# Below this, tesseract's own orientation guess is not worth having and the
# four rotations are tried properly instead.
OSD_MIN_CONFIDENCE = 1.5
_OSD_ROTATE = re.compile(r"Rotate:\s*(\d+)")
_OSD_CONF = re.compile(r"Orientation confidence:\s*([\d.]+)")


@dataclass
class Variant:
    """One rendering of the page, and what was done to it."""
    name: str
    image: np.ndarray


@dataclass
class Page:
    path: Path
    rotation: int            # degrees applied to get it upright
    osd_confidence: float
    variants: list[Variant]
    rotation_source: str = "osd"
    # The upright photo in colour, before any binarisation. A vision model
    # is trained on real photos, not on the black-and-white renderings built
    # for tesseract, so it gets this instead of page.base.
    color: np.ndarray | None = None

    @property
    def base(self) -> np.ndarray:
        return self.variants[0].image


def _load_upright(path: Path) -> np.ndarray:
    """Load the photo with its EXIF orientation already applied."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def detect_rotation(image: np.ndarray) -> tuple[int, float]:
    """How far the text is turned, via tesseract's orientation detection.

    Returns (degrees to rotate clockwise to make it upright, confidence).
    A failure is (0, 0.0): never guess a rotation, because rotating a page
    that was already upright destroys it.
    """
    small = _resize_to_width(image, OSD_WIDTH)
    ok, buf = cv2.imencode(".png", small)
    if not ok:
        return 0, 0.0
    try:
        out = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "0"],
            input=buf.tobytes(), capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired):
        return 0, 0.0
    rotate = _OSD_ROTATE.search(out)
    conf = _OSD_CONF.search(out)
    return (int(rotate.group(1)) % 360 if rotate else 0,
            float(conf.group(1)) if conf else 0.0)


def _legibility(image: np.ndarray) -> float:
    """How much readable text a quick OCR pass finds. Used only to choose
    between the four rotations when OSD is unsure; the absolute value means
    nothing, the comparison does."""
    small = _resize_to_width(image, 1200)
    ok, buf = cv2.imencode(".png", small)
    if not ok:
        return 0.0
    try:
        out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"],
                             input=buf.tobytes(), capture_output=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    text = out.stdout.decode("utf-8", "replace")
    # Words of three or more letters: gibberish from an upside-down page
    # produces plenty of characters but very few of these.
    return float(len(re.findall(r"[A-Za-z]{3,}", text)))


def choose_rotation(image: np.ndarray) -> tuple[int, float, str]:
    """Which way is up. Returns (degrees, confidence, how it was decided).

    Tesseract's orientation detection is trusted when it is confident. When
    it is not - which is exactly the case on a sideways thermal receipt - all
    four rotations are read and the most legible one wins. Guessing here is
    expensive: rotating an upright page destroys it, and leaving a sideways
    one alone loses the bill completely.
    """
    rotation, confidence = detect_rotation(image)
    if confidence >= OSD_MIN_CONFIDENCE:
        return rotation, confidence, "osd"
    scores = {deg: _legibility(_rotate(image, deg)) for deg in (0, 90, 180, 270)}
    best = max(scores, key=lambda d: scores[d])
    runner_up = max((s for d, s in scores.items() if d != best), default=0.0)
    if scores[best] < 5 or scores[best] < runner_up * 1.3:
        return rotation, confidence, "osd (unsure, no clear alternative)"
    return best, confidence, f"legibility {scores[best]:.0f} vs {runner_up:.0f}"


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=interp)


def _fit(image: np.ndarray) -> np.ndarray:
    """Upscale small photos, and cap very tall ones (one real screenshot is
    6616 px tall, nearly all of it menu upsell)."""
    h, w = image.shape[:2]
    if w < TARGET_WIDTH:
        image = _resize_to_width(image, TARGET_WIDTH)
        h, w = image.shape[:2]
    if h * w > MAX_PIXELS:
        scale = (MAX_PIXELS / (h * w)) ** 0.5
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def deskew(gray: np.ndarray, limit: float = 12.0) -> np.ndarray:
    """Straighten a page tilted by a few degrees in the hand.

    Only small angles are corrected: a large one means the text mask caught
    the table edge or a shadow instead of the print, and acting on that would
    tilt a straight page.
    """
    inverted = cv2.bitwise_not(gray)
    threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(threshold)
    if coords is None or len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > limit:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _flatten_lighting(gray: np.ndarray) -> np.ndarray:
    """Divide out the slow brightness gradient a phone photo leaves behind,
    so one corner of a thermal receipt is not darker than the other."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape) / 30)
    flat = cv2.divide(gray, background, scale=255)
    return flat.astype(np.uint8)


def _sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    """Sauvola thresholding: local mean and deviation, which holds up on
    faint thermal print where a global threshold loses whole lines."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True)
    sq = cv2.boxFilter(g * g, -1, (window, window), normalize=True)
    std = cv2.sqrt(cv2.max(sq - mean * mean, 0))
    threshold = mean * (1 + k * ((std / 128.0) - 1))
    return np.where(g > threshold, 255, 0).astype(np.uint8)


def build_variants(gray: np.ndarray) -> list[Variant]:
    """The renderings handed to the OCR engines, cheapest first."""
    flat = _flatten_lighting(gray)
    sharp = cv2.filter2D(flat, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    return [
        Variant("gray", gray),
        Variant("flat", flat),
        Variant("otsu", otsu),
        Variant("adaptive", adaptive),
        Variant("sauvola", _sauvola(flat)),
    ]


def prepare(path: str | Path) -> Page:
    """Photo in, upright renderings out."""
    path = Path(path)
    image = _load_upright(path)
    rotation, confidence, how = choose_rotation(image)
    image = _rotate(image, rotation)
    image = _fit(image)
    gray = deskew(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return Page(path=path, rotation=rotation, osd_confidence=confidence,
                variants=build_variants(gray), rotation_source=how, color=image)
