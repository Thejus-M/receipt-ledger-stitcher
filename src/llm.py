"""A local LLM, as one more candidate method — never a requirement.

This is methods C4/C5 of the plan's Stage C bake-off: a text model cleaning
up the OCR output, and a vision model reading the photo directly. Both are
measured in the same `bench` table as the deterministic methods, on the same
ten bills, and neither becomes the default unless it actually wins there.

The one hard rule, non-negotiable: **nothing talks to anything but this
machine.** `OLLAMA_URL` is a literal loopback address, not read from an
environment variable, so a stray `OLLAMA_HOST` in the shell can never point
this at a network endpoint. The single network action anywhere in this
feature is the one-off `ollama pull` of the model weights — a download, not
an upload — and that happens from the command line, never from this module.

Everything here degrades to "this method is unavailable" rather than
raising: a bench run must still produce a table for the methods that don't
need an LLM when ollama is not running.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

# Deliberately not "localhost" (which can resolve over the network on a
# misconfigured system) and deliberately not read from the environment.
OLLAMA_URL = "http://127.0.0.1:11434"
CONNECT_TIMEOUT = 5

TEXT_MODEL = "qwen2.5:3b"
VISION_MODEL = "granite3.2-vision:2b"
VISION_MODEL_ALT = "qwen2.5vl:3b"
# Google's efficient on-device multimodal line ("effective 2B" params,
# despite a larger download — MatFormer lets it run a smaller slice of a
# bigger network at inference time). User-suggested candidate; benched
# like every other method rather than trusted on "it works great" alone.
VISION_MODEL_GEMMA = "gemma3n:e2b"

SCHEMA_HINT = """Reply with nothing but a single JSON object, no markdown fences, no commentary,
with exactly these keys (use null for anything you cannot read; never guess a value):

{
  "merchant": string or null,
  "date": "YYYY-MM-DD" or null,
  "time": "HH:MM" or null,
  "gstin": string or null,
  "invoice_no": string or null,
  "total": number or null,
  "subtotal": number or null,
  "discount": number or null,
  "round_off": number or null,
  "payment_hint": one of "UPI","CARD","CASH" or null,
  "is_cart": true or false,
  "items": [{"description": string, "qty": number or null, "unit": string or null,
             "rate": number or null, "hsn": string or null, "amount": number}],
  "taxes": [{"rate": number, "taxable": number, "cgst": number, "sgst": number,
            "igst": number, "cess": number}]
}

"qty" and "rate" are the quantity and per-unit price printed on an itemised
line (e.g. "2 x 45.00") — null when the bill only prints one amount for the
line. "unit" is the printed unit (kg, pc, ltr, ...) when one is shown.

Money is always a plain number of rupees, e.g. 456.00, never a string, never with a
rupee sign. "is_cart" is true only for an app's checkout screen (an unpaid cart with
a "Place Order" or "Pay" button, not yet charged), never for a paid receipt."""

# Built with a plain placeholder and .replace(), not str.format(): the
# schema above is full of its own { } braces, which .format() would try to
# read as substitutions and fail on immediately.
TEXT_PROMPT = """The text below was OCR'd from a photographed Indian shop bill. OCR drops
decimal points and confuses similar characters, so use the bill's own arithmetic
(items should sum to the subtotal; subtotal minus discount plus/minus round-off
should equal the total) to decide between readings rather than taking any single
number as certainly correct.

OCR TEXT:
---
<<<OCR_TEXT>>>
---

""" + SCHEMA_HINT

VISION_PROMPT = """This is a photo of an Indian shop bill (a GST retail bill, a restaurant
invoice, a liquor-shop bill, or a phone screenshot of an order/checkout screen).
Read it directly from the image.

""" + SCHEMA_HINT


def available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=CONNECT_TIMEOUT)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def has_model(name: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=CONNECT_TIMEOUT) as r:
            tags = {m["name"] for m in json.loads(r.read())["models"]}
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError):
        return False
    return name in tags or any(t.split(":")[0] == name.split(":")[0] for t in tags)


def _post(path: str, payload: dict, timeout: int) -> dict | None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict:
    """The JSON object inside a model's reply.

    Instruction-following is imperfect even with `format: "json"`: a model
    still sometimes wraps the object in a markdown fence or a sentence of
    preamble. The outermost {...} is taken and parsed; a failure returns {}
    rather than raising, since a bad reply must degrade to "nothing read",
    not crash the bench.
    """
    if not text:
        return {}
    stripped = text.strip()
    for candidate in (stripped, *(m.group(0) for m in [_JSON_OBJECT.search(stripped)] if m)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            continue
    return {}


def generate_json(model: str, prompt: str, images: list[bytes] | None = None,
                  timeout: int = 180, on_chunk=None) -> dict:
    """One call, JSON out. {} on any failure: no model, no connection, no
    parseable reply.

    `on_chunk`, when given, switches to ollama's streaming reply (one NDJSON
    line per token/fragment) and is called with each text fragment as it
    arrives - the only way to show a caller anything while a vision model
    is still thinking, since the plain path otherwise blocks silently for
    however long the model takes. The final parsed JSON is the same either
    way: streaming changes nothing but when the text becomes visible.
    """
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": on_chunk is not None,
              # num_predict bounds how many tokens the model can generate.
              # Without it, a reply measured in minutes on this CPU turned
              # out to run long enough to blow through a 380s timeout with
              # nothing returned - the compact JSON this schema asks for
              # needs nowhere near that many tokens.
              "options": {"temperature": 0, "num_predict": 700}}
    if images:
        payload["images"] = [base64.b64encode(b).decode("ascii") for b in images]
    if on_chunk is None:
        result = _post("/api/generate", payload, timeout)
        if not result:
            return {}
        return extract_json(result.get("response", ""))
    return extract_json(_stream(payload, timeout, on_chunk))


def _stream(payload: dict, timeout: int, on_chunk) -> str:
    """The streaming twin of `_post`: reads NDJSON lines as ollama sends
    them instead of waiting for one complete body, calling `on_chunk` with
    each text fragment. Degrades to an empty reply on any failure, same as
    the non-streaming path."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    pieces: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                piece = obj.get("response", "")
                if piece:
                    pieces.append(piece)
                    on_chunk(piece)
                if obj.get("done"):
                    break
    except (urllib.error.URLError, OSError, TimeoutError):
        return ""
    return "".join(pieces)


def _encode(image: np.ndarray, max_width: int = 1200) -> bytes:
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode the page for the vision model")
    return buf.tobytes()


def structure_from_text(raw_text: str, model: str = TEXT_MODEL, timeout: int = 90,
                        on_chunk=None) -> dict:
    """Hand already-OCR'd text to a text model and ask it to structure and
    reconcile the figures. Cheaper than vision, but only as good as the OCR
    underneath it."""
    prompt = TEXT_PROMPT.replace("<<<OCR_TEXT>>>", raw_text[:6000])
    return generate_json(model, prompt, timeout=timeout, on_chunk=on_chunk)


def structure_from_image(color_image: np.ndarray, model: str = VISION_MODEL,
                         timeout: int = 180, on_chunk=None) -> dict:
    """Hand the photo itself to a vision model. Slower, and not bounded by
    whatever OCR already got wrong."""
    return generate_json(model, VISION_PROMPT, images=[_encode(color_image)], timeout=timeout,
                         on_chunk=on_chunk)
