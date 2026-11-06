"""Read-only bridge to the Transactions widget's database.

The widget owns `txdb`; this project only borrows it, exactly as the bank
fetchers do (`TXDB_HOME` points at that repo). Nothing here writes: the
reports are meant to be run and judged before anything is filed.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

TXDB_HOME = Path(os.environ.get("TXDB_HOME", "/home/pc/Work/omarchy-transactions"))
DB_PATH = Path(os.environ.get("TXDB_DB", TXDB_HOME / "data/transactions.db"))

if str(TXDB_HOME) not in sys.path:
    sys.path.insert(0, str(TXDB_HOME))


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"No transactions database at {DB_PATH}. "
                         "Set TXDB_HOME or TXDB_DB to point at the widget's repo.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def transactions(conn: sqlite3.Connection) -> list[dict]:
    """Every row, in the shape match.py scores: amount in rupees, negative
    for money out, with the account and whether it is already tracked."""
    rows = conn.execute("""
        SELECT t.id, t.date, t.merchant, t.counterparty, t.amount_paise,
               a.bank, a.account_type, a.last4,
               (k.transaction_id IS NOT NULL) AS tracked
        FROM transactions t
        JOIN accounts a ON a.id = t.account_id
        LEFT JOIN tracked k ON k.transaction_id = t.id
        ORDER BY t.date DESC, t.seq DESC""").fetchall()
    return [{
        "id": r["id"], "date": r["date"], "merchant": r["merchant"],
        "counterparty": r["counterparty"], "amount": r["amount_paise"] / 100.0,
        "bank": r["bank"], "account_type": r["account_type"], "last4": r["last4"],
        "tracked": bool(r["tracked"]),
    } for r in rows]
