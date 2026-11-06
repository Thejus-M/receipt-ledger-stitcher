"""receipt-ledger-stitcher — tie photographed bills to ledger rows.

    uv run python -m src.main bench        [--methods A,E] [--only kunnil]
    uv run python -m src.main match-report [--window N | --all-history]
    uv run python -m src.main read  <photo> [--method E-consensus]

`bench` measures every method against the hand-transcribed ground truth in
tests/fixtures/receipts, field by field, with the seconds each one costs.

`match-report` takes each bill and scans the whole transaction history for
it, printing the ranked candidates with the reasons behind each score.

Both write nothing. Their job is to say which approach actually works on
these bills, so the defaults are chosen on evidence.

Nothing in this program talks to the network.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

from . import match as M
from . import store
from .parse import ALL_METHODS, LLM_METHODS, METHODS, parse_file

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures/receipts"
PHOTOS = Path(os.environ.get("RECEIPTS_DIR", "/run/media/pc/MY Drive/KDE Share"))
MONEY_TOLERANCE = 0.05


def load_truth() -> list[dict]:
    return [json.loads(Path(f).read_text("utf-8")) for f in sorted(glob.glob(str(FIXTURES / "*.json")))]


def _money_ok(got, want) -> bool:
    if want is None:
        return got is None
    return got is not None and abs(float(got) - float(want)) <= MONEY_TOLERANCE


def _merchant_ok(got, want) -> bool:
    if not want:
        return True
    if not got:
        return False
    return fuzz.partial_ratio(str(got).upper(), str(want).upper()) >= 70


def compare(parse, truth: dict) -> dict[str, bool | None]:
    """Field by field. None means the bill does not print that field, so the
    method is neither credited nor blamed for it."""
    want_date = truth.get("date") or truth.get("exif_date")
    out: dict[str, bool | None] = {
        "total": _money_ok(parse.total, truth.get("total")),
        "date": parse.date == want_date,
        "merchant": _merchant_ok(parse.merchant, truth.get("merchant")),
        "subtotal": _money_ok(parse.subtotal, truth["subtotal"]) if truth.get("subtotal") is not None else None,
        "tax": _money_ok(parse.tax_total, truth["tax_total"]) if truth.get("tax_total") is not None else None,
        "items": (len(parse.items) == truth.get("item_count")) if truth.get("items_confident") else None,
        "gstin": (parse.gstin == truth["gstin"]) if truth.get("gstin") else None,
    }
    return out


def bench(methods: list[str], only: str | None) -> int:
    truth = load_truth()
    if only:
        truth = [t for t in truth if only.lower() in t["file"].lower()
                 or only.lower() in (t.get("merchant") or "").lower()]
    if not truth:
        print("no fixtures matched", file=sys.stderr)
        return 1

    fields = ["total", "date", "merchant", "subtotal", "tax", "items", "gstin"]
    results: dict[str, dict] = {}
    for method in methods:
        hits = {f: [0, 0] for f in fields}        # [correct, applicable]
        seconds = 0.0
        per_bill = []
        for t in truth:
            photo = PHOTOS / t["file"]
            if not photo.exists():
                print(f"  missing photo: {photo}", file=sys.stderr)
                continue
            started = time.monotonic()
            parse, _ = parse_file(photo, method)
            seconds += time.monotonic() - started
            marks = compare(parse, t)
            for f, ok in marks.items():
                if ok is None:
                    continue
                hits[f][1] += 1
                hits[f][0] += int(bool(ok))
            per_bill.append((t, parse, marks))
            print(f"  {method:14} {t['merchant'][:22]:24} "
                  f"total={'OK ' if marks['total'] else 'BAD'} "
                  f"got={parse.total if parse.total is not None else '-'!s:>9} "
                  f"want={t['total']:>9.2f}  conf={parse.confidence:>3}  "
                  f"{','.join(parse.checks_passed) or 'no checks passed'}", flush=True)
        results[method] = {"hits": hits, "seconds": seconds, "n": len(per_bill), "bills": per_bill}

    print("\n" + "=" * 86)
    print("ACCURACY BY METHOD  (correct / bills where the field applies)")
    print("=" * 86)
    header = f"{'method':14}" + "".join(f"{f:>10}" for f in fields) + f"{'sec/bill':>10}"
    print(header)
    print("-" * len(header))
    for method, data in results.items():
        row = f"{method:14}"
        for f in fields:
            correct, applicable = data["hits"][f]
            row += f"{f'{correct}/{applicable}':>10}" if applicable else f"{'-':>10}"
        n = max(data["n"], 1)
        row += f"{data['seconds'] / n:>10.1f}"
        print(row)

    # Which bill each method gets right, not just how many.
    print("\n" + "=" * 86)
    print("PER BILL  (. = correct total, X = wrong, - = unreadable)")
    print("=" * 86)
    names = list(results)
    print(f"{'bill':34}{'kind':20}" + "".join(f"{m.split('-')[0]:>6}" for m in names))
    print("-" * (54 + 6 * len(names)))
    for t in truth:
        row = f"{t['merchant'][:32]:34}{t['kind'][:18]:20}"
        for m in names:
            hit = next((b for b in results[m]["bills"] if b[0]["file"] == t["file"]), None)
            if hit is None:
                row += f"{'?':>6}"
            elif hit[2]["total"]:
                row += f"{'.':>6}"
            else:
                row += f"{'-' if hit[1].total is None else 'X':>6}"
        print(row)

    print("\nBills no method could total:")
    unresolved = []
    for t in truth:
        if not any(any(m[0]["file"] == t["file"] and m[2]["total"] for m in d["bills"])
                   for d in results.values()):
            unresolved.append(t)
    for t in unresolved:
        print(f"  {t['merchant'][:30]:32} {t['total']:>9.2f}  {t.get('notes','')[:60]}")
    if not unresolved:
        print("  (none)")
    return 0


def match_report(window: int | None, method: str, only: str | None) -> int:
    """Every bill against the whole ledger, ranked, with reasons."""
    truth = load_truth()
    if only:
        truth = [t for t in truth if only.lower() in t["file"].lower()
                 or only.lower() in (t.get("merchant") or "").lower()]
    conn = store.connect()
    txns = store.transactions(conn)
    spends = [t for t in txns if t["amount"] < 0]
    print(f"{len(txns)} transactions in the ledger ({len(spends)} spends), "
          f"{sum(1 for t in txns if not t['tracked'])} untracked.")
    print(f"Window: {'the whole history' if window is None else f'+/-{window} days'}. "
          f"Nothing is written.\n")

    auto = exact = none = 0
    for t in truth:
        photo = PHOTOS / t["file"]
        if not photo.exists():
            continue
        parse, _ = parse_file(photo, method)
        # The report judges matching, not OCR, so it falls back to the
        # transcribed truth when a method misreads the total.
        receipt = {"merchant": parse.merchant or t.get("merchant"),
                   "date": parse.date or t.get("date") or t.get("exif_date"),
                   "total": parse.total if parse.total is not None else t.get("total")}
        read_total = "" if _money_ok(parse.total, t.get("total")) else \
            f"   [OCR read the total as {parse.total}; using {receipt['total']:.2f}]"
        cart = "  [cart, may never have been paid]" if not t.get("is_receipt", True) else ""
        print(f"{t['merchant'][:30]:32} {receipt['date']}  Rs {receipt['total']:>9.2f}{cart}{read_total}")

        ranked = M.rank(receipt, txns, max_days=window, limit=4)
        if not ranked:
            none += 1
            print("     --  nothing plausible in the whole history")
        for c in ranked:
            txn = c.transaction
            flag = "tracked  " if txn["tracked"] else "UNTRACKED"
            mark = "<<" if c is ranked[0] and M.auto_pick(ranked) is c else "  "
            print(f"   {c.score:>3} {mark} {txn['date']}  Rs {abs(txn['amount']):>9.2f}  "
                  f"{txn['bank']:5} {str(txn['merchant'])[:34]:36} {flag}")
            print(f"            {' . '.join(c.reasons)}")
        picked = M.auto_pick(ranked)
        if picked:
            auto += 1
        if ranked and ranked[0].score >= 60 and "exact amount" in ranked[0].reasons:
            exact += 1
        print()

    print("=" * 92)
    print(f"{len(truth)} bills: {auto} would be auto-filed, {exact} have an exact-amount "
          f"candidate, {none} have nothing at all.")
    print("Auto-filing needs a score of "
          f"{M.AUTO_FLOOR}+ and a {M.AUTO_MARGIN}-point lead over the runner-up, "
          "and in the widget it is further restricted to untracked rows.")
    return 0


def scan_cmd(folder: str, method: str, apply: bool, reparse: bool) -> int:
    from . import ingest
    from pathlib import Path as _Path

    # Printed live, not just summarised at the end: a scan of several
    # photos used to go quiet until every single one had been read, which
    # for an LLM method is minutes with no sign anything was happening.
    model = LLM_METHODS[method][1] if method in LLM_METHODS else None
    chunk_state = {"started": False}

    def on_start(photo):
        chunk_state["started"] = False
        print(f"\n--- {photo.name} ---", flush=True)
        if not model:
            print("Reading with OCR ...", flush=True)

    def on_chunk(piece):
        if not chunk_state["started"]:
            print(f"Calling {model} ...\n", flush=True)
            chunk_state["started"] = True
        print(piece, end="", flush=True)

    entries = ingest.scan(_Path(folder), method=method, apply=apply, reparse=reparse,
                          on_start=on_start, on_chunk=(on_chunk if model else None))
    if not entries:
        print("Nothing new to read.")
        return 0
    print("\n\n--- results ---\n")
    filed = 0
    for e in entries:
        p = e.parse
        flags = []
        if p.is_cart:
            flags.append("checkout capture")
        if p.needs_review:
            flags.append(f"check this parse ({p.confidence}%)")
        print(f"{(p.merchant or '?')[:30]:32} {p.date or '????-??-??'}  "
              f"Rs {p.total if p.total is not None else 0:>9.2f}  "
              f"{len(p.items)} items" + (f"   [{', '.join(flags)}]" if flags else ""))
        for w in p.warnings:
            print(f"      ! {w}")
        if e.linked_to:
            filed += 1
            print(f"      -> filed against {e.linked_to} ({e.score}): {' . '.join(e.reasons or [])}")
        else:
            print("      -> needs a transaction")
    print(f"\n{len(entries)} bills read, {filed} filed automatically"
          + ("  (dry run: nothing was written)" if not apply else ""))
    return 0


def read_one(path: str, method: str) -> int:
    parse, seconds = parse_file(path, method)
    print(json.dumps({
        "merchant": parse.merchant, "date": parse.date, "time": parse.time,
        "gstin": parse.gstin, "invoice_no": parse.invoice_no,
        "total": parse.total, "subtotal": parse.subtotal, "discount": parse.discount,
        "round_off": parse.round_off, "tax_total": parse.tax_total,
        "cgst": parse.cgst, "sgst": parse.sgst,
        "payment_hint": parse.payment_hint, "is_cart": parse.is_cart,
        "rotation": parse.rotation, "confidence": parse.confidence,
        "needs_review": parse.needs_review,
        "checks_passed": parse.checks_passed, "warnings": parse.warnings,
        "items": parse.items, "seconds": round(seconds, 1),
    }, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stitch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="measure every method against the ground truth")
    b.add_argument("--methods", default=",".join(METHODS),
                   help=f"comma-separated; deterministic: {', '.join(METHODS)}; "
                        f"local LLM (needs ollama running - see src/llm.py): "
                        f"{', '.join(LLM_METHODS)}")
    b.add_argument("--only", help="only bills whose filename or merchant contains this")

    m = sub.add_parser("match-report", help="rank every bill against the whole ledger")
    m.add_argument("--window", type=int, default=None,
                   help="limit candidates to +/- N days (default: the whole history)")
    m.add_argument("--method", default="E-consensus", choices=list(ALL_METHODS))
    m.add_argument("--only", help="only bills whose filename or merchant contains this")

    sc = sub.add_parser("scan", help="import new photos and file the obvious ones")
    sc.add_argument("--folder", default=str(PHOTOS))
    sc.add_argument("--method", default="H-vision-llm-gemma", choices=list(ALL_METHODS))
    sc.add_argument("--dry-run", action="store_true",
                    help="do all the work and print it, writing nothing")
    sc.add_argument("--reparse", action="store_true", help="re-read photos already imported")

    rp = sub.add_parser("reparse-one", help="re-read one already-imported receipt by id")
    rp.add_argument("receipt_id")
    rp.add_argument("--method", default="H-vision-llm-gemma", choices=list(ALL_METHODS))

    r = sub.add_parser("read", help="parse one photo and print the result")
    r.add_argument("photo")
    r.add_argument("--method", default="H-vision-llm-gemma", choices=list(ALL_METHODS))

    args = parser.parse_args(argv)
    if args.command == "bench":
        names = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in names if m not in ALL_METHODS]
        if unknown:
            print(f"unknown method(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        return bench(names, args.only)
    if args.command == "scan":
        return scan_cmd(args.folder, args.method, not args.dry_run, args.reparse)
    if args.command == "reparse-one":
        from . import ingest
        model = LLM_METHODS[args.method][1] if args.method in LLM_METHODS else None
        # "Calling {model} ..." prints lazily, on the first streamed token,
        # not up front: real work happens before the model is ever
        # contacted (loading the photo, normalising it, a quick OCR pass
        # for the anchor check), and saying "Calling" during that would be
        # describing a step that hasn't started yet.
        print("Loading photo ...", flush=True)
        chunk_state = {"started": False}

        def on_chunk(piece):
            if not chunk_state["started"]:
                print(f"\nCalling {model} ...\n", flush=True)
                chunk_state["started"] = True
            print(piece, end="", flush=True)

        parse = ingest.reparse_receipt(args.receipt_id, args.method,
                                       on_chunk=(on_chunk if model else None))
        print("\n\n--- done ---\n")
        print(f"{(parse.merchant or '?')[:30]:32} {parse.date or '????-??-??'}  "
              f"Rs {parse.total if parse.total is not None else 0:>9.2f}  "
              f"{len(parse.items)} items"
              + (f"   [check this parse ({parse.confidence}%)]" if parse.needs_review else ""))
        for w in parse.warnings:
            print(f"      ! {w}")
        return 0
    if args.command == "match-report":
        return match_report(args.window, args.method, args.only)
    if args.command == "read":
        return read_one(args.photo, args.method)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
