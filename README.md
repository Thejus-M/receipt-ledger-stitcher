# receipt-ledger-stitcher

An extension to the **home.transactions** widget, not a project in its own
right. The widget knows what each purchase *cost*; this stitches on what was
actually *bought*, by reading the photographed bill and tying it to the bank
row that paid for it.

Reads photographed shop bills **on this machine** and files them against the
transactions in the Transactions widget's database.

Photos arrive in `/run/media/pc/MY Drive/KDE Share` from the phone. This
project turns each one into a merchant, a date, a grand total, a GST
breakdown and a list of items, then finds the bank transaction that paid for
it.

**Nothing leaves the device.** No cloud OCR, no API, no telemetry. If a local
LLM is enabled later it runs on `127.0.0.1` through ollama, and the only
network action in the whole project is the one-off model download.

## Setup

```sh
uv venv
uv pip install -r requirements.txt --torch-backend=cpu
```

The `--torch-backend=cpu` matters: easyocr pulls torch, and this machine has
Intel graphics, so the default resolution downloads ~2.5 GB of CUDA wheels it
can never use.

Needs `tesseract` (with `eng` and `osd`) on the PATH.

## Commands

```sh
.venv/bin/python -m src.main bench                  # accuracy of every method
.venv/bin/python -m src.main match-report           # bills vs the whole ledger
.venv/bin/python -m src.main read <photo>           # parse one photo
```

`bench` and `match-report` write nothing. They exist so the defaults — which
OCR method, how wide a date window, how high a score must be before a bill is
filed automatically — are chosen from measurements on real bills rather than
from a guess.

## How it reads a bill

A photographed thermal receipt defeats any single OCR pass, so the pipeline
is built around that rather than around one engine.

1. **`normalize.py`** — EXIF orientation, then tesseract's own orientation
   detection to turn the page upright (two of the ten real bills are
   sideways and upside down, and return nothing without it), deskew, upscale,
   then several renderings of the same page: grayscale, lighting-flattened,
   Otsu, adaptive, Sauvola.
2. **`engines.py`** — tesseract at several page-segmentation modes and
   EasyOCR, all returning the same `Word` boxes, so fields can be found by
   geometry and not only by text.
3. **`fields.py`** — labels, dates, totals, tax lines and item rows. It never
   commits to one reading of a number: thermal print loses decimal points, so
   `461` is offered as 4.61, 46.1 and 461 together.
4. **`validate.py`** — the referee. It picks the combination of readings that
   makes the bill agree with itself:

   ```
   items sum                      == item total
   item total - discount +/- round off == grand total
   taxable + CGST + SGST          == grand total
   ```

   This is the heart of the project. On a real bill one rendering read the
   discount as `481` and another as `461`; only 4.61 makes
   `460.82 - 4.61 - 0.21` come to the printed `456.00`, so the arithmetic
   decides and nobody has to guess. A reading is also rewarded for using
   every figure the bill printed, because dropping a line and landing within
   a paise by luck is not the same as reading it right.
5. **`match.py`** — scores a bill against transactions: amount first (the one
   field OCR gets right), then date distance, then merchant words. A name
   that clearly disagrees **subtracts**, because a ₹250 restaurant bill and a
   ₹250 cinema ticket three days apart are otherwise indistinguishable.

## What it refuses to do

- **Guess a date.** An unreadable date is `null` with a warning. A wrong date
  hides the correct transaction from the matcher, which is worse than no date.
- **Believe the word "cash".** Real bills print `GPAY 456.00` and
  `Tender Cash 456.00` together, and another prints `To: CASH` above
  `Card : 149.00`. A named rail always beats the word.
- **Assume a checkout capture is not a bill.** Two of the ten photos are
  checkout screens, photographed before paying because the receipt issued
  afterwards drops the line items. They are bills, and are parsed as such;
  the capture is only noted because a last-step tip or coupon can move the
  amount actually paid.
- **Auto-file on a weak match.** A link needs a high score and a clear lead
  over the runner-up, and every link is reversible.

## The local-LLM bake-off (Stage C of the plan)

Two more methods, run through `ollama` on this machine only:

- `F-text-llm` — a text model (`qwen2.5:3b`) reconciles already-OCR'd text.
- `G-vision-llm` — a vision model (`granite3.2-vision:2b`, picked for its
  document/table focus) reads the photo directly.
- `G2-vision-llm-alt` — `qwen2.5vl:3b`, a stronger fallback if the 2B vision
  model underperforms.

```sh
ollama pull granite3.2-vision:2b
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:3b
.venv/bin/python -m src.main bench --methods E-consensus,F-text-llm,G-vision-llm,G2-vision-llm-alt
```

`src/llm.py` talks to `http://127.0.0.1:11434` only — a literal constant, never
read from the environment, so nothing can repoint it at a network endpoint.
The only network action anywhere in this is the `ollama pull` above, run from
the command line, never from this code. Neither LLM method becomes the
default; they are measured in the same table as every OCR method and only
replace one if `bench` shows they are worth their seconds.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

No OCR, no network: text and numbers in, decisions out. The cases come from
the ten real bills, and several exist because the first implementation got
them wrong.

`tests/fixtures/receipts/` holds the ground truth, transcribed by eye. It
states only what is legible — an unreadable field is `null`, and a block that
cannot be read to the last digit carries `"confident": false` so the bench
scores the totals and skips the shaky part instead of punishing a method for
disagreeing with a bad transcription.
