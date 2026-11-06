"""Scoring a bill against the transactions in the ledger.

Deliberately stdlib-only, so the same rules can live in the widget's
`txdb/` package (which takes no third-party dependency) and be tested there.

The weights come from what the ten real bills actually did:

* **Amount carries the most.** It is the one field OCR reads reliably on a
  thermal photo, and a bank debit matches a bill's grand total to the paise.
* **The merchant can veto.** A real near-miss: a ₹250 restaurant bill and a
  ₹250 PVR Inox cinema ticket three days apart. Amount and date alone rank
  the cinema as a good match; only the name says otherwise. So a name that
  clearly disagrees subtracts rather than merely failing to add.
* **Dates drift by a day or two.** A card settles later than the purchase,
  so a one-day gap is normal and must not be punished like a week.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

# Transaction text is mostly rails and reference numbers; these are the words
# that never identify a shop.
NOISE_TOKENS = {
    "UPI", "P2M", "P2A", "P2V", "DR", "CR", "NEFT", "IMPS", "RTGS", "ACH", "NACH", "ECS",
    "BANK", "LTD", "LIMITED", "PVT", "PRIVATE", "INDIA", "COM", "THE", "AND", "PAY",
    "PAYMENT", "PAYMENTS", "TRANSFER", "TO", "FROM", "BRN", "POS", "ATM", "REF", "TXN",
    "STORE", "SHOP", "RETAIL", "TRIVANDRUM", "KERALA", "IN", "OF", "AT",
    # Bank names ride along in every UPI string and identify the rail, not
    # the shop: "UPI/P2M/165729881322/HDFC BANK" names no merchant at all.
    "HDFC", "ICICI", "AXIS", "SBI", "KOTAK", "YES", "IDFC", "INDUSIND", "PNB",
    "CANARA", "UNION", "FEDERAL", "RBL", "AU", "BOB", "PAYTM", "GPAY", "PHONEPE",
}
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def tokens(text: str | None) -> set[str]:
    """Identifying words of a shop name, with the rails stripped out."""
    if not text:
        return set()
    out = set()
    for raw in _SPLIT.split(str(text).upper()):
        token = raw.strip()
        if len(token) < 3 or token in NOISE_TOKENS or token.isdigit():
            continue
        if len(token) > 12 and any(c.isdigit() for c in token):
            continue                      # reference strings
        out.add(token)
    return out


def _days_between(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    try:
        return abs((date.fromisoformat(a[:10]) - date.fromisoformat(b[:10])).days)
    except ValueError:
        return None


@dataclass
class Candidate:
    transaction: dict
    score: int
    reasons: list[str]


def score(receipt: dict, txn: dict, *, max_days: int | None = 5) -> tuple[int, list[str]]:
    """0-100 and why. 0 means disqualified, never "merely poor"."""
    reasons: list[str] = []

    amount = txn.get("amount")
    if amount is None or amount >= 0:
        return 0, ["not a spend"]
    spent = round(abs(float(amount)), 2)
    total = receipt.get("total")
    if total is None:
        return 0, ["the bill has no total to match on"]
    total = round(float(total), 2)

    gap = abs(spent - total)
    if gap <= 0.011:
        points, reasons = 60, ["exact amount"]
    elif gap <= 1.0:
        points, reasons = 55, [f"amount off by {gap:.2f} (rounding)"]
    elif gap <= max(5.0, total * 0.01):
        points, reasons = 40, [f"amount off by {gap:.2f}"]
    else:
        return 0, [f"amount off by {gap:.2f}"]

    days = _days_between(receipt.get("date"), txn.get("date"))
    if days is None:
        reasons.append("no date on the bill")
    elif max_days is not None and days > max_days:
        return 0, [f"{days} days apart"]
    elif days == 0:
        points += 25
        reasons.append("same day")
    elif days == 1:
        points += 18
        reasons.append("1 day apart")
    elif days <= 3:
        points += 8
        reasons.append(f"{days} days apart")
    elif days <= 5:
        points += 2
        reasons.append(f"{days} days apart")
    else:
        reasons.append(f"{days} days apart")

    bill_words = tokens(receipt.get("merchant"))
    txn_words = tokens(txn.get("merchant")) | tokens(txn.get("counterparty"))
    if bill_words and txn_words:
        shared = bill_words & txn_words
        if shared:
            points += min(15, 6 + 3 * len(shared))
            reasons.append(f"merchant '{'/'.join(sorted(shared)[:2])}' matches")
        elif _partial_overlap(bill_words, txn_words):
            points += 6
            reasons.append("merchant partly matches")
        else:
            points -= 15
            reasons.append("merchant does not match")
    return max(0, min(points, 100)), reasons


def _partial_overlap(a: set[str], b: set[str]) -> bool:
    """One name contained in the other: 'KSBC' inside 'KSBCFL01'."""
    for x in a:
        for y in b:
            if len(x) >= 4 and len(y) >= 4 and (x in y or y in x):
                return True
    return False


def rank(receipt: dict, transactions: list[dict], *, max_days: int | None = 5,
         untracked_only: bool = False, limit: int = 10) -> list[Candidate]:
    """Best candidates first. `max_days=None` searches the whole history,
    which is what the accuracy report wants; the shipped auto-link keeps the
    window."""
    out: list[Candidate] = []
    for txn in transactions:
        if untracked_only and txn.get("tracked"):
            continue
        points, reasons = score(receipt, txn, max_days=max_days)
        if points > 0:
            out.append(Candidate(transaction=txn, score=points, reasons=reasons))
    out.sort(key=lambda c: (-c.score, _days_between(receipt.get("date"), c.transaction.get("date")) or 999))
    return out[:limit]


AUTO_FLOOR = 85
AUTO_MARGIN = 15


def auto_pick(candidates: list[Candidate]) -> Candidate | None:
    """The one case confident enough to file without being asked: a clear
    winner, well clear of the runner-up. A wrong auto-link is worse than
    none, so this stays strict."""
    if not candidates or candidates[0].score < AUTO_FLOOR:
        return None
    if len(candidates) > 1 and candidates[0].score - candidates[1].score < AUTO_MARGIN:
        return None
    return candidates[0]


def parse_iso(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None
