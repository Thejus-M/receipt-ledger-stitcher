"""The arithmetic referee.

OCR hands up several readings of every figure; this module decides between
them by asking the bill to agree with itself. A real example from the ten:
one rendering read a discount as 481, another as 461, and only one of those
makes 460.82 - discount - 0.21 come to the printed 456.00. That is a fact
about the bill, not a preference, so it is the right way to choose.

The identities, in the order they carry weight:

    items sum            == item total
    item total - discount +/- round off == grand total
    taxable + CGST + SGST (+ IGST + cess) == grand total
    items sum            == grand total          (bills with no subtotal)

Everything is done in paise as integers: 0.1 + 0.2 in floats is not 0.3, and
a bill that balances must balance exactly.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

TOLERANCE_PAISE = 100          # a rupee: bills round their own totals
NEAR_PAISE = 5                 # what counts as an exact hit


def paise(value: float | None) -> int | None:
    return None if value is None else int(round(value * 100))


def rupees(value: int | None) -> float | None:
    return None if value is None else round(value / 100, 2)


@dataclass
class Parse:
    """One complete reading of a bill."""
    merchant: str | None = None
    date: str | None = None
    time: str | None = None
    gstin: str | None = None
    invoice_no: str | None = None
    total: float | None = None
    subtotal: float | None = None
    discount: float | None = None
    round_off: float | None = None
    tax_total: float | None = None
    cgst: float | None = None
    sgst: float | None = None
    igst: float | None = None
    cess: float | None = None
    items: list[dict] = field(default_factory=list)
    payment_hint: str | None = None
    is_cart: bool = False
    rotation: int = 0
    method: str = ""
    raw_text: str = ""
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: int = 0

    @property
    def needs_review(self) -> bool:
        return self.confidence < 70 or self.total is None


def _close(a: int | None, b: int | None, tolerance: int = TOLERANCE_PAISE) -> bool:
    return a is not None and b is not None and abs(a - b) <= tolerance


def _agreement(a: int | None, b: int | None, exact: int, approximate: int) -> int:
    """Points for two figures agreeing: full marks for landing on the paise,
    fewer for landing within a rupee. Without the distinction, a bill whose
    discount reads 4.81 'balances' as well as the true 4.61, since both come
    within the rounding tolerance - which is exactly the choice the referee
    is here to make."""
    if a is None or b is None:
        return 0
    gap = abs(a - b)
    if gap <= NEAR_PAISE:
        return exact
    return approximate if gap <= TOLERANCE_PAISE else 0


def choose_item_amounts(candidate_lists: list[list[float]], target: int | None) -> list[float]:
    """Pick one amount per item so they sum to `target` if any combination can.

    Walks the reachable sums rather than the combinations: fifteen items with
    three readings each is fourteen million combinations but only a few
    thousand distinct sums, and the first exact hit is the answer.
    """
    firsts = [c[0] if c else 0.0 for c in candidate_lists]
    if target is None or not candidate_lists:
        return firsts
    reachable: dict[int, list[float]] = {0: []}
    ceiling = target + 5000
    for candidates in candidate_lists:
        nxt: dict[int, list[float]] = {}
        for total, chosen in reachable.items():
            for value in (candidates or [0.0])[:4]:
                new = total + (paise(value) or 0)
                if new > ceiling or new in nxt:
                    continue
                nxt[new] = chosen + [value]
        if not nxt:
            return firsts
        reachable = nxt
        if len(reachable) > 60000:                      # keep the nearest half
            keep = sorted(reachable, key=lambda s: abs(s - target))[:30000]
            reachable = {s: reachable[s] for s in keep}
    for tolerance in (0, NEAR_PAISE, TOLERANCE_PAISE):
        for total in sorted(reachable, key=lambda s: abs(s - target)):
            if abs(total - target) <= tolerance:
                return reachable[total]
    return firsts


def referee(candidates: dict, items_by_layout: dict[str, list[list[dict]]],
            anchor: float | None = None) -> Parse:
    """Choose the reading of each figure that makes the bill add up.

    `candidates` holds lists of possible values per field, best-guess first;
    `items_by_layout` holds the item rows each table layout produced.
    `anchor` is the largest amount actually printed with a decimal point or a
    rupee mark anywhere on the bill.

    The anchor exists because a scale error is self-consistent: if every
    figure is read a hundred times too small, the items still sum to the
    subtotal and the tax still adds up, and a 2,550 rupee bill was filed as
    2.50. Nothing inside the arithmetic can catch that - only a figure the
    bill printed in full can.
    """
    floor = (anchor * 0.5) if anchor else None
    best: tuple[int, Parse] | None = None
    totals = list(candidates.get("total") or [])
    unlabelled_total = False
    if not totals and anchor:
        # No line said "total", but the bill printed an amount in full. It is
        # very likely the total, and saying so with a warning beats throwing
        # the whole bill away - which is what used to happen, taking the
        # subtotal and tax down with it even though both had been read.
        totals = [anchor]
        unlabelled_total = True
    if not totals:
        totals = [None]
    subtotals = (candidates.get("subtotal") or []) + [None]
    discounts = (candidates.get("discount") or []) + [None]
    round_offs = (candidates.get("round_off") or []) + [None]

    for total, subtotal, discount, round_off in itertools.product(
            totals[:5], subtotals[:5], discounts[:4], round_offs[:4]):
        t, s, d, r = (paise(total), paise(subtotal), paise(discount), paise(round_off))
        if t is None or t <= 0:
            continue
        if floor is not None and total < floor:
            continue        # the bill prints a larger figure than this total
        if s is not None and (s <= 0 or s > t * 12):
            continue
        if d is not None and s is not None and d > s:
            continue

        passed: list[str] = []
        score = 0

        # item total - discount +/- round off == grand total
        if s is not None:
            earned = max(_agreement(s - (d or 0) + sign * (r or 0), t, 40, 22)
                         for sign in (1, -1))
            if earned:
                passed.append("subtotal-discount-roundoff=total")
                score += earned

        # taxable + tax == grand total
        tax = paise(_first(candidates.get("tax_total")))
        cgst = paise(_first(candidates.get("cgst")))
        sgst = paise(_first(candidates.get("sgst")))
        igst = paise(_first(candidates.get("igst")))
        cess = paise(_first(candidates.get("cess")))
        tax_sum = sum(v for v in (cgst, sgst, igst, cess) if v) or tax
        if s is not None and tax_sum:
            earned = _agreement(s + tax_sum, t, 30, 16)
            if earned:
                passed.append("taxable+tax=total")
                score += earned

        # The items themselves. Every candidate row-set from every reading
        # is tried, under every layout it was read as; the one whose rows
        # actually sum to the target wins, which is what stops a noisier
        # reading's extra (wrong) rows from crowding out a cleaner one that
        # had fewer rows but got them right.
        target = s if s is not None else t
        layout_best, layout_name, layout_delta, layout_rows = [], "", None, []
        for name, row_sets in items_by_layout.items():
            for rows in row_sets:
                chosen = choose_item_amounts([r["amount_candidates"] for r in rows], target)
                delta = abs(sum(paise(v) or 0 for v in chosen) - (target or 0))
                if layout_delta is None or delta < layout_delta:
                    layout_best, layout_name, layout_delta, layout_rows = chosen, name, delta, rows
        if layout_delta is not None and layout_delta <= TOLERANCE_PAISE:
            passed.append(f"items-sum={'subtotal' if s is not None else 'total'}")
            score += 25 if layout_delta <= NEAR_PAISE else 15

        # A reading should explain what the bill actually prints. Without
        # this, "460.82 - 4.81 = 456.01" scored as well as the true
        # "460.82 - 4.61 - 0.21 = 456.00", because dropping the printed
        # round-off line happened to land within a paise. Using a figure the
        # bill went to the trouble of printing is evidence for the reading.
        if s is not None:
            score += 4
        if d is not None and candidates.get("discount"):
            score += 3
        if r is not None and candidates.get("round_off"):
            score += 3
        # What the engine actually read, before any decimal was reinstated,
        # is the reading to beat.
        if candidates.get("total") and total == candidates["total"][0]:
            score += 12
        if candidates.get("subtotal") and subtotal is not None and subtotal == candidates["subtotal"][0]:
            score += 5

        if best is None or score > best[0]:
            parse = Parse(total=total, subtotal=subtotal, discount=discount, round_off=round_off,
                          tax_total=rupees(tax_sum) if tax_sum else None,
                          cgst=rupees(cgst), sgst=rupees(sgst), igst=rupees(igst), cess=rupees(cess),
                          checks_passed=passed)
            parse.items = [{"description": row["description"], "amount": amount,
                           **_extra_item_fields(row)}
                           for row, amount in zip(layout_rows, layout_best)]
            best = (score, parse)

    if best is None:
        # Nothing balanced, but whatever was read is still worth keeping.
        parse = Parse(
            total=_first(candidates.get("total")),
            subtotal=_first(candidates.get("subtotal")),
            discount=_first(candidates.get("discount")),
            round_off=_first(candidates.get("round_off")),
            cgst=_first(candidates.get("cgst")), sgst=_first(candidates.get("sgst")),
            igst=_first(candidates.get("igst")), cess=_first(candidates.get("cess")),
            tax_total=_first(candidates.get("tax_total")),
            warnings=["no grand total could be read"])
        all_row_sets = [rows for row_sets in items_by_layout.values() for rows in row_sets]
        rows = max(all_row_sets, key=len, default=[])
        parse.items = [{"description": r["description"],
                        "amount": (r["amount_candidates"] or [0.0])[0],
                        **_extra_item_fields(r)} for r in rows]
        if parse.cgst or parse.sgst:
            parse.tax_total = round(sum(v for v in (parse.cgst, parse.sgst, parse.igst,
                                                    parse.cess) if v), 2) or parse.tax_total
        _check_gst_breakdown(parse, candidates)
        return parse
    parse = best[1]
    if unlabelled_total and parse.total is not None:
        parse.warnings.append(
            f"no line was labelled a total; used the largest printed amount ({parse.total:.2f})")
    # A tax total nobody printed, but whose parts were read.
    if parse.tax_total is None:
        parts = [v for v in (parse.cgst, parse.sgst, parse.igst, parse.cess) if v]
        if parts:
            parse.tax_total = round(sum(parts), 2)
    _check_gst_breakdown(parse, candidates)
    return parse


def _first(values):
    return values[0] if values else None


def _extra_item_fields(row: dict) -> dict:
    """qty/unit/rate/hsn, when a reading (so far only the LLM path) gave
    them. Deterministic OCR rows carry none of these keys, so this is a
    no-op there."""
    return {k: row[k] for k in ("qty", "unit", "rate", "hsn") if row.get(k) is not None}


def _check_gst_breakdown(parse: Parse, candidates: dict) -> None:
    """Flag a tax split that disagrees with the bill's own total.

    Checked against the *raw* subtotal/total candidates, not whichever
    reading `referee` ended up choosing — a plausible-looking split (cgst
    5.00 + sgst 9.00 = 14.00) can fail every other check badly enough that
    `referee` drops the subtotal reading entirely rather than pick it, which
    would otherwise hide the exact contradiction this exists to name. Real
    case: gemma3n:e2b on the Cake World bill read total 149.00, subtotal
    141.90, cgst 5.00, sgst 9.00 — 14.00 of tax where the bill's own figures
    imply 7.10. Only checked when individual slabs were actually given: a
    bill with no CGST/SGST split at all must not be penalised for lacking
    one it never printed.
    """
    slabs = [v for v in (parse.cgst, parse.sgst, parse.igst, parse.cess) if v]
    if not slabs:
        return
    subtotal = _first(candidates.get("subtotal"))
    total = _first(candidates.get("total")) if candidates.get("total") else parse.total
    if subtotal is None or total is None:
        return
    slab_sum = sum(paise(v) or 0 for v in slabs)
    implied = (paise(total) or 0) - (paise(subtotal) or 0)
    if abs(slab_sum - implied) > TOLERANCE_PAISE:
        parse.warnings.append(
            f"the GST breakdown ({rupees(slab_sum):.2f}) doesn't match the bill's own "
            f"total minus subtotal ({rupees(implied):.2f})")


def score_confidence(parse: Parse, ocr_confidence: float) -> int:
    """0-100. Arithmetic that holds is worth far more than an engine's own
    confidence, which is high on a cleanly misread digit."""
    if parse.total is None:
        return 0
    score = 35
    score += 25 if "subtotal-discount-roundoff=total" in parse.checks_passed else 0
    score += 20 if "taxable+tax=total" in parse.checks_passed else 0
    score += 15 if any(c.startswith("items-sum") for c in parse.checks_passed) else 0
    if parse.date:
        score += 5
    if parse.merchant:
        score += 5
    score = min(score, 100)
    return int(round(score * (0.75 + 0.25 * min(max(ocr_confidence, 0.0), 1.0))))


def add_warnings(parse: Parse) -> None:
    """Say plainly what did not add up, so a bill is never filed as certain
    when it is not."""
    if parse.total is None:
        parse.warnings.append("no grand total could be read")
    if parse.items and parse.total is not None:
        items_sum = sum(paise(i["amount"]) or 0 for i in parse.items)
        target = paise(parse.subtotal) if parse.subtotal is not None else paise(parse.total)
        if target and abs(items_sum - target) > TOLERANCE_PAISE:
            parse.warnings.append(
                f"items add up to {rupees(items_sum):.2f}, the bill says {rupees(target):.2f}")
    if parse.subtotal is not None and parse.total is not None and not parse.checks_passed:
        parse.warnings.append("none of the bill's own totals agree with each other")
    if parse.date is None:
        parse.warnings.append("no date could be read; the photo's timestamp is used instead")
    if parse.is_cart:
        parse.warnings.append(
            "captured at the checkout screen; the amount paid may differ if a tip "
            "or coupon was added after this")
