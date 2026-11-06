# Where this stands

## Done

**Stage 1 — reading bills** (`~/Work/receipt-ledger-stitcher`)

Full deterministic pipeline (`normalize`, `engines`, `fields`, `validate`,
`match`, `parse`, `ingest`, `main`). Ground truth transcribed for all ten
bills. **63 tests pass** (38 logic/OCR + 25 for the LLM path below).

Project renamed to `receipt-ledger-stitcher` and moved; `~/Work/receipt-reader`
no longer exists.

**Stage 2 — storage** (`~/Work/omarchy-transactions`)

Migration v11 (`receipts`, `receipt_items`, `receipt_taxes`, `receipt_links`,
`receipts_view`), `txdb/receipts.py`, six new `txdb` subcommands, `receipts`
key in the snapshot. Live database migrated, all 435 transactions intact.
**133 plugin tests pass.**

## Clean OCR bench (post item-merge-fix, final for this round)

| Method | Total | Date | Merchant | Subtotal | Tax | Items | GSTIN | sec/bill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-tess6 | 8/10 | 7/10 | 5/10 | 0/9 | 1/9 | 5/9 | 2/6 | 8.7 |
| B-tess4 | 7/10 | 7/10 | 7/10 | 1/9 | 1/9 | 6/9 | 2/6 | 9.6 |
| C-tess6-bin | 4/10 | 7/10 | 4/10 | 0/9 | 1/9 | 4/9 | 4/6 | 10.8 |
| D-easyocr | 5/10 | 6/10 | 6/10 | 1/9 | 1/9 | 3/9 | 1/6 | 31.1 |
| E-consensus | 9/10 | 7/10 | 5/10 | 1/9 | 2/9 | 4/9 | 5/6 | 27.0 |

Only KSRTC (a redBus screenshot with no price printed on it at all) defeats
every method. E-consensus is the strongest baseline so far.

## In progress — the local-LLM bake-off (Stage C of the plan)

Added on request: `src/llm.py` (ollama client, hard-locked to
`http://127.0.0.1:11434`, never read from the environment), wired into
`parse.py` as two more bench-able methods:

- F-text-llm — qwen2.5:3b reconciles already-OCR'd text.
- G-vision-llm — granite3.2-vision:2b reads the photo directly.
- G2-vision-llm-alt — qwen2.5vl:3b, the stronger fallback.

Both go through the same referee() as every OCR method, so they are scored
on the same arithmetic, not a separate rule. 25 tests added, all mocked at
the network boundary - no model call is ever a test dependency.

ollama serve is running (user-level, not the systemd unit - that needs a
password this session doesn't have interactively; started directly instead,
confirmed bound to 127.0.0.1 only via ss). All three models are downloading
now (ollama pull, the one network action in this whole feature). ~2-4 GB
each; not finished as of this note.

## Next

1. Once the pulls finish: bench --methods E-consensus,F-text-llm,G-vision-llm,G2-vision-llm-alt
   and report that table.
2. match-report against the whole ledger.
3. scan --dry-run.
4. Stage 3: the RECEIPTS section in Panel.qml.
