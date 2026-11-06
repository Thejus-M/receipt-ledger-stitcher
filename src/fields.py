"""Turning OCR output into the fields of a bill.

Two things make this harder than it looks, and both are visible in the ten
real bills:

1. **Numbers lose their decimal point.** A real Kunnil bill's "4.61" came back
   as "481" and "461" from two different renderings. So this module never
   commits to a single reading: `money_candidates` offers every plausible
   interpretation of a token and validate.py picks the combination that makes
   the bill's own arithmetic work.

2. **Item tables come in two shapes.** Most bills put one item on one line
   ("Loaded Fries Large 1 250 250.00"). Kunnil splits each item across two:
   the description and HSN on one line, then MRP/qty/rate/GST/total on the
   next. Both are extracted and the one whose amounts sum to the printed item
   total wins.

Labels are matched at the start of a line or after a separator, never
mid-sentence, so "Looking for a higher credit limit?" is not a credit limit
and "Statement Details - Generated on 01 NOV 2026" still is a date.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .engines import Word, line_text, words_to_lines

# A figure is money only with a currency mark or two decimals; without that
# rule the "2026" of "05 NOV 2026" reads as a 2,026 rupee amount.
MONEY_RE = re.compile(
    r"(?:[-+]\s*)?(?:₹|Rs\.?|INR)\s*(?:[-+]\s*)?[\d,]+(?:\.\d{1,2})?"
    r"|(?:[-+]\s*)?[\d,]+\.\d{2}", re.I)
# A bare number, for item tables where the engine dropped the decimal point.
NUMBER_RE = re.compile(r"[-+]?[\d,]+(?:\.\d+)?")
LOOSE_MONEY_RE = re.compile(
    r"(?:[-+]\s*)?(?:₹|Rs\.?|INR)\s*(?:[-+]\s*)?[\d,]+(?:\.\d{1,2})?"
    r"|(?:[-+]\s*)?[\d,]+\.\d{1,2}"
    r"|(?<![\d.])[-+]?\d{1,5}(?![\d.])", re.I)
DATE_RE = re.compile(
    r"\d{1,2}[\s/.-](?:\d{1,2}|[A-Za-z]{3,9})[,\s/.-]+\d{2,4}|\d{4}-\d{2}-\d{2}")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)?\b", re.I)
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][\dA-Z]Z[\dA-Z])\b")
INVOICE_RE = re.compile(
    r"\b(?:bill|invoice|inv|receipt)\b\.?\s*(?:no|number|#)?\.?\s*[:#-]\s*"
    r"([A-Z0-9][A-Z0-9/_-]{3,})", re.I)

DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%d %b %y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y",
                "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y")

# The generic "total" must not swallow "Sub Total", "Item Total" or
# "Total Qty": on one bill that made the subtotal 238.10 the grand total and
# on another it made the item count the amount.
_NOT_A_GRAND_TOTAL = r"(?<!sub )(?<!item )(?<!qnt )"
TOTAL_LABELS = (r"grand total", r"g\.?\s*total(?:\(rnd\))?", r"net (?:payable|amount)",
                r"amount payable", r"to pay", r"total amount",
                _NOT_A_GRAND_TOTAL + r"total\s*(?:rs\.?|₹)(?!\s*qty)",
                _NOT_A_GRAND_TOTAL + r"total(?!\s*(?:qty|qnt|quantity|items?)\b)")
SUBTOTAL_LABELS = (r"item total(?: amount)?", r"sub\s*total", r"taxable (?:value|amt|amount)",
                   r"net amt", r"total qnt \d+ nrt-?val(?:ue|ur)")
DISCOUNT_LABELS = (r"bill discount", r"discount", r"extra discount(?: for you)?", r"you saved")
ROUNDOFF_LABELS = (r"round\s*off", r"round\s*o[fl]+", r"rounding")
CGST_LABELS = (r"cgst(?:\s*amt)?(?:@[\d.]+)?",)
SGST_LABELS = (r"sgst(?:\s*amt)?(?:@[\d.]+)?",)
IGST_LABELS = (r"igst(?:\s*amt)?(?:@[\d.]+)?",)
CESS_LABELS = (r"cess(?:\s*amt)?",)
TAX_LABELS = (r"gst (?:&|and) other charges", r"gst amount", r"gst amt", r"tax amount", r"total tax")
# Payment hints, in the order they beat each other: a named rail wins over
# the word "cash", because bills print "Tender Cash" beside "GPAY".
PAYMENT_PATTERNS = (
    ("UPI", r"\b(upi|gpay|google pay|phonepe|paytm|bhim|paid via upi)\b"),
    ("CARD", r"\b(card|visa|mastercard|rupay|credit card|debit card)\b"),
    ("ICICI", r"\bicici\b"), ("HDFC", r"\bhdfc\b"), ("AXIS", r"\baxis\b"), ("SBI", r"\bsbi\b"),
    ("CASH", r"\b(cash|tender cash)\b"),
)
# Words that mean this is a shopping cart, not a purchase.
CART_MARKERS = (r"place order", r"clear cart", r"add more items", r"apply coupon",
                r"pay\s*₹?\s*[\d,]+", r"add items", r"cooking requests", r"complete your meal",
                r"delivery tip", r"add tip")
MERCHANT_STOP = re.compile(
    r"gst|invoice|bill\s*(no|to)|date|phone|ph:|mob|fssai|tel|www\.|@|^to\s*:|counter|cashier", re.I)
NOISE_LINE = re.compile(r"^[\W_]*$")


@dataclass
class Doc:
    """One OCR reading of a page."""
    words: list[Word]
    lines: list[list[Word]] = field(default_factory=list)
    text: str = ""

    @classmethod
    def build(cls, words: list[Word]) -> "Doc":
        lines = words_to_lines(words)
        return cls(words=words, lines=lines,
                   text="\n".join(line_text(line) for line in lines))

    @property
    def line_texts(self) -> list[str]:
        return [line_text(line) for line in self.lines]


def clean(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


def money_value(token: str) -> float | None:
    """The number in a money token, unsigned."""
    match = NUMBER_RE.search(clean(token).replace("₹", " ").replace(",", ""))
    if not match:
        return None
    try:
        return abs(float(match.group()))
    except ValueError:
        return None


def money_candidates(token: str) -> list[float]:
    """Every plausible reading of a money token, best guess first.

    OCR drops decimal points on thermal print, so "461" may be 4.61, 46.1 or
    461. All are offered; the arithmetic referee decides. Tokens that already
    carry two decimals are taken at face value.
    """
    text = clean(token).replace("₹", "").replace(",", "").strip()
    text = re.sub(r"(?i)^(rs\.?|inr)\s*", "", text).strip()
    negative = text.startswith("-")
    digits = re.sub(r"[^\d.]", "", text)
    if not digits:
        return []
    out: list[float] = []

    def add(value: float) -> None:
        value = round(value, 2)
        if value not in out:
            out.append(value)

    if "." in digits:
        try:
            add(float(digits))
        except ValueError:
            pass
        # A misplaced point is common too: "4.81" vs "48.1".
        bare = digits.replace(".", "")
        if bare.isdigit() and len(bare) >= 3:
            add(int(bare) / 100)
    elif digits.isdigit():
        add(float(digits))
        if len(digits) >= 3:
            add(int(digits) / 100)
        if len(digits) >= 2:
            add(int(digits) / 10)
    return [-v for v in out] if negative else out


def parse_date(text: str) -> str | None:
    cleaned = re.sub(r"\bSept\b", "Sep", clean(text), flags=re.I).replace(",", " ")
    cleaned = " ".join(cleaned.split())
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        if parsed.year < 100:
            parsed = parsed.replace(year=2000 + parsed.year)
        # A bill dated far in the future is a misread, not a bill.
        if 2000 <= parsed.year <= datetime.now().year + 1:
            return parsed.strftime("%Y-%m-%d")
    return None


def _label_at(line: str, pattern: str) -> re.Match | None:
    """The label must open the line or follow a separator, never sit
    mid-sentence."""
    return re.search(rf"(?:^|[-–—:|·•]\s*){pattern}\b\s*[:.\-]?\s*", line, re.I)


ANY_LABEL = re.compile(
    r"(?i)(?:^|[-–—:|·•]\s*)(grand total|g\.?\s*total|item total|sub\s*total|total|discount|"
    r"round\s*o|taxable|net amt|net amount|cgst|sgst|igst|cess|gst|amount payable|to pay|"
    r"you saved|balance|tender|change)\b")


def _first_amount(text: str, token_re: re.Pattern) -> str | None:
    """The first figure in `text` that is an amount rather than a rate.

    Tax lines print the rate beside the money - "CGST@2.5 2.5% 9.05" - and
    taking the first number there read a 5.95 rupee tax as 2.5.
    """
    for match in token_re.finditer(text):
        after = text[match.end():match.end() + 1]
        if after == "%":
            continue
        if re.fullmatch(r"\s*%", text[match.end():match.end() + 2] or ""):
            continue
        return match.group(0)
    return None


def _carries_a_label(line: str) -> bool:
    """Does this line introduce a figure of its own?"""
    return bool(ANY_LABEL.search(clean(line)))


def labelled_tokens(lines: list[str], patterns, token_re: re.Pattern,
                    loose_re: re.Pattern | None = None) -> list[str]:
    """Tokens printed with a label: on its own line, the next two, or the one
    above (some bills print the amount above its caption).

    The label's own line also accepts a bare integer, because thermal print
    loses decimal points and "Bill Discount : 481" means 4.61. Adjacent lines
    do not: a loose match there would wander off and pick up the next
    figure on the bill, which is how a discount once came back as the grand
    total."""
    found: list[str] = []
    for pattern in patterns:
        for i, line in enumerate(lines):
            match = _label_at(line, pattern)
            if not match:
                continue
            same_line = line[match.end():]
            hit = _first_amount(same_line, loose_re or token_re)
            if hit:
                found.append(hit)
                continue
            for candidate in (*lines[i + 1:i + 3], lines[i - 1] if i else ""):
                if _carries_a_label(candidate):
                    continue          # that figure belongs to its own caption
                hit = _first_amount(candidate, token_re)
                if hit:
                    found.append(hit)
                    break
    return found


def find_total(lines: list[str]) -> list[float]:
    """Candidate grand totals, best first. The largest labelled figure is
    preferred: a bill's grand total is its biggest number far more often than
    not, and 'Total Qty: 1' is excluded by the money rule."""
    out: list[float] = []
    for token in labelled_tokens(lines, TOTAL_LABELS, MONEY_RE, LOOSE_MONEY_RE):
        for value in money_candidates(token):
            if value > 0 and value not in out:
                out.append(value)
    return out


def find_labelled_value(lines: list[str], patterns) -> list[float]:
    out: list[float] = []
    for token in labelled_tokens(lines, patterns, MONEY_RE, LOOSE_MONEY_RE):
        for value in money_candidates(token):
            if value not in out:
                out.append(value)
    return out


def find_date(lines: list[str]) -> str | None:
    for line in lines:
        if re.search(r"\b(date|dt|dated|on)\b", line, re.I):
            for match in DATE_RE.finditer(line):
                parsed = parse_date(match.group(0))
                if parsed:
                    return parsed
    for line in lines:                      # any date, if none was labelled
        for match in DATE_RE.finditer(line):
            parsed = parse_date(match.group(0))
            if parsed:
                return parsed
    return None


def find_time(text: str) -> str | None:
    match = TIME_RE.search(text)
    if not match:
        return None
    hour, minute, meridiem = int(match.group(1)), match.group(2), (match.group(3) or "").upper()
    if meridiem == "PM" and hour < 12:
        hour += 12
    if meridiem == "AM" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}" if 0 <= hour <= 23 else None


def find_gstin(text: str) -> str | None:
    match = GSTIN_RE.search(text.upper().replace(" ", ""))
    return match.group(1) if match else None


def find_invoice_no(lines: list[str]) -> str | None:
    for line in lines:
        match = INVOICE_RE.search(line)
        if match:
            value = match.group(1).strip(" .:-")
            if not re.fullmatch(r"(no|number|date|to|cash)", value, re.I) and len(value) >= 4:
                return value
    return None


def find_merchant(lines: list[str]) -> str | None:
    """The shop's name: the first substantial line before the bill's
    machinery (GST number, invoice number, date, phone) begins."""
    for line in lines[:12]:
        text = clean(line)
        if len(text) < 3 or NOISE_LINE.match(text) or MERCHANT_STOP.search(text):
            continue
        letters = sum(c.isalpha() for c in text)
        if letters < 3 or letters < len(text) * 0.5:
            continue
        if re.fullmatch(r"(?i)(tax )?invoice|bill|receipt|cash memo|retail invoice", text):
            continue
        return text.upper()
    return None


def find_payment_hint(text: str) -> str | None:
    """What paid for it, if the bill says. 'Cash' is only believed when no
    other rail is named: real bills print 'Tender Cash' beside 'GPAY'."""
    hits = [name for name, pattern in PAYMENT_PATTERNS if re.search(pattern, text, re.I)]
    for name in hits:
        if name != "CASH":
            return name
    return "CASH" if hits else None


def looks_like_cart(text: str) -> bool:
    """Was this captured at the checkout screen rather than after payment?

    Not a reason to reject it. The receipt a shop app issues after payment
    usually drops the line items, so the cart is deliberately where the
    detail is photographed - these are bills. It is still worth recording,
    because the amount finally paid can differ from the cart total if a tip
    or a coupon is applied at the last step, which matters when matching.
    """
    return sum(1 for pattern in CART_MARKERS if re.search(pattern, text, re.I)) >= 2


def find_tax_slabs(lines: list[str]) -> list[dict]:
    """GST slab rows, as printed. Only what is labelled is read; the referee
    checks them against the totals afterwards."""
    slabs: list[dict] = []
    cgst = find_labelled_value(lines, CGST_LABELS)
    sgst = find_labelled_value(lines, SGST_LABELS)
    igst = find_labelled_value(lines, IGST_LABELS)
    cess = find_labelled_value(lines, CESS_LABELS)
    if cgst or sgst or igst or cess:
        slabs.append({"cgst": cgst[0] if cgst else None, "sgst": sgst[0] if sgst else None,
                      "igst": igst[0] if igst else None, "cess": cess[0] if cess else None})
    return slabs


# --- item tables ------------------------------------------------------------

_INDEX = re.compile(r"^\s*(\d{1,2})[).\s]")
_HSN = re.compile(r"\b\d{6,8}\b")
# Thermal print plus OCR turns digits into these letters constantly: a real
# item row came back as "25000 1000 22872 SO0 Q00 2018", where SO0 is 5.00
# and Q00 is 0.00. Without undoing that, the row does not look numeric at all
# and its item is lost.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                                   "S": "5", "s": "5", "I": "1", "l": "1",
                                   "|": "1", "B": "8", "Z": "2", "G": "6"})
ITEM_STOP = re.compile(
    r"(?i)\b(total|subtotal|sub total|discount|round|gst|cgst|sgst|igst|cess|qty|quantity|"
    r"balance|tender|change|savings?|you saved|amount|date|time|bill|invoice|cashier|"
    r"counter|points?|net|taxable|item description|hsn|mrp|price|rate)\b")


def _digitize(text: str) -> str:
    """Undo the digit/letter confusions, for deciding what a row is."""
    return text.translate(_DIGIT_LOOKALIKES)


def _numbers_in(line: str) -> list[str]:
    return [m.group(0) for m in NUMBER_RE.finditer(line.replace(",", ""))]


def _is_amount(token: str) -> bool:
    """Could this token be a money amount at the end of an item row?

    Six- and eight-digit runs are HSN codes, which sit in exactly that
    position on a Kunnil bill, and ten-digit runs are phone numbers.
    """
    digits = token.replace(",", "").replace("-", "")
    if "." in digits:
        return True
    return digits.isdigit() and 1 <= len(digits) <= 5


def _is_numeric_line(line: str) -> bool:
    """A row of figures: the second half of a two-line item.

    Measured by the share of digits after undoing the letter confusions,
    rather than by "contains no letters", which no real thermal scan passes.
    """
    text = _digitize(clean(line))
    body = re.sub(r"[\s.,%₹/-]", "", text)
    if len(body) < 3 or len(_numbers_in(text)) < 2:
        return False
    return sum(c.isdigit() for c in body) / len(body) >= 0.75


def _describes_an_item(text: str) -> bool:
    description = clean(_HSN.sub("", re.sub(_INDEX, "", text)))
    letters = sum(c.isalpha() for c in description)
    return letters >= 4 and not ITEM_STOP.search(description)


# Items always sit above the totals block; nothing below "Item Total" or
# "Grand Total" is merchandise. This is what keeps the GST summary table out
# of the item list, which no shape-based rule managed.
_TOTALS_START = re.compile(
    r"(?i)(?:^|[-\s:|])(item total|sub\s*total|grand total|g\.?\s*total|total\s*(?:rs|₹|:)|"
    r"taxable value|total qty|total amount|net amount|amount payable|to pay)")


def item_region(lines: list[str]) -> list[str]:
    """The slice of the bill that can hold items: everything above the first
    totals line. Returns all of it when no totals line is recognised."""
    for i, line in enumerate(lines):
        if _TOTALS_START.search(clean(line)):
            return lines[:i] if i >= 2 else lines
    return lines


def _tail_is_numeric(text: str, first_number: str) -> bool:
    """The figures must END the line: a description with a number in the
    middle ('An ISO 9001:2015 Certified Company') is not an item row."""
    at = text.find(first_number)
    if at < 0:
        return False
    tail = _digitize(text[at:])
    body = re.sub(r"[\s.,%₹/:-]", "", tail)
    return bool(body) and sum(c.isdigit() for c in body) / len(body) >= 0.8


def _clean_description(text: str) -> str:
    return clean(_HSN.sub("", re.sub(_INDEX, "", text))).upper()


def items_single_line(lines: list[str]) -> list[dict]:
    """One item per line: description then its figures."""
    items: list[dict] = []
    for line in item_region(lines):
        text = clean(line)
        if not text or _is_numeric_line(text) or not _describes_an_item(text):
            continue
        numbers = _numbers_in(_digitize(text))
        if len(numbers) < 2 or not _is_amount(numbers[-1]) or not _tail_is_numeric(text, numbers[0]):
            continue
        head = text[:text.find(numbers[0])] if numbers[0] in text else text
        description = _clean_description(head)
        if len(description) < 3:
            continue
        items.append({"description": description,
                      "amount_candidates": money_candidates(numbers[-1]),
                      "numbers": numbers})
    return items


def items_two_line(lines: list[str]) -> list[dict]:
    """Kunnil's layout: the description (with its HSN code) on one line and
    the figures on the next.

    Driven from the figures backwards, because OCR splits a printed row into
    two or three text lines as often as not ("FANDOARK WHITE 1000 1 BOS0N0"
    then "i TOBLERONE"): the description is whatever legible text sits just
    above the numbers.
    """
    items: list[dict] = []
    region = item_region(lines)
    for i, line in enumerate(region):
        if not _is_numeric_line(line):
            continue
        numbers = _numbers_in(_digitize(clean(line)))
        if len(numbers) < 3 or not _is_amount(numbers[-1]):
            continue
        parts: list[str] = []
        for back in range(1, 4):                    # look a little way up
            if i - back < 0:
                break
            above = clean(region[i - back])
            if _is_numeric_line(above):
                break
            if _describes_an_item(above):
                parts.insert(0, _clean_description(above))
            elif parts:
                break
        description = clean(" ".join(parts))
        if len(description) < 3:
            continue
        items.append({"description": description,
                      "amount_candidates": money_candidates(numbers[-1]),
                      "numbers": numbers})
    return items
