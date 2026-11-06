# Getting a bill photographed well enough to read

Every tip below traces back to a real failure found while building this, not a
guess. Each cites the bill that proved it.

## Photos of paper receipts

1. **Fill the frame, and don't let anything crop the edges.** The KSBC liquor
   bill's right-hand figures (715.1_, 1794.9_, 40.0_) were sliced off by the
   shop's own printed side-strip inside the photo's frame. Leave a small
   margin around the whole receipt — nothing printed should touch the edge of
   the shot.
2. **Shoot square-on, not at an angle.** A few degrees of tilt is corrected
   automatically; a sharp angle is not, and it also makes the bill's own
   columns (MRP / Qty / Rate / Amount) overlap in the image, which is exactly
   where OCR loses the decimal point.
3. **Get the whole bill the right way up in the photo**, even though the code
   can now detect and fix sideways or upside-down shots. Two of the ten bills
   here were rotated 90° and 180° respectively, and getting that detection
   right took real work — a straight shot just skips that risk entirely.
4. **Flatten a curled receipt before shooting.** Thermal paper curls at the
   ends, which is usually where the total sits. A curled edge reads as a
   blur, not a mis-read number — there's nothing arithmetic can recover from
   that.
5. **Even light, no glare.** A shadow across the paper, or a camera-flash
   hotspot on glossy thermal stock, both defeat the thresholding step. Diffuse
   daylight or indoor light without a direct bulb overhead works best.
6. **Fresh print beats old print.** Thermal receipts fade with heat and time;
   a faded one looks like a weak-but-readable gray smear that OCR guesses at
   rather than reads. Photograph it soon after the purchase if it matters.

## Screenshots of app bills (Swiggy, redBus, etc.)

7. **Don't let one giant scrolling screenshot bury the total.** The Kottayam
   Kitchen capture was 6,616 pixels tall — a full scroll of menu upsells —
   with the actual "To Pay ₹139" squeezed into the last sixth. Both OCR and
   the LLM lost it in the noise. If the app lets you, screenshot just the
   items section and, separately, the final totals section, rather than one
   continuous scroll.
8. **Photograph the cart before paying, as you've been doing** — that's
   correct, since the post-payment receipt usually drops the line items. Just
   make sure the **totals block is visible in that same screenshot**, not
   scrolled past it.
9. **A screenshot with no price at all defeats every method tried.** The
   redBus ticket here showed the journey and the seat, but its fare only
   appeared on a different screen; neither OCR nor an LLM can read a number
   that isn't in the image. If the bill has an order-confirmation screen with
   the amount paid, that's the one to capture.

## What the pipeline already compensates for, so these matter less than they
used to, but still help

- A slightly duplicated or truncated number: the parser cross-checks every
  figure against the bill's own arithmetic (items sum to the subtotal,
  subtotal minus discount equals the total) and prefers whichever reading
  actually balances.
- A missing year, a sideways photo, or a dropped decimal point: all handled
  automatically now. They still cost accuracy at the margins, which is why
  the tips above are worth the extra two seconds when taking the photo.
