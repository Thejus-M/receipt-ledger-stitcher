# Ground truth

One file per bill photo, transcribed by eye from the image. This is the yardstick the
bench measures every OCR/LLM method against, so it says only what is actually legible:
a field nobody can read is `null`, never a guess, and a block that is legible but not
to the last digit carries `"confident": false` so the bench scores the totals and
skips the shaky part rather than punishing a method for disagreeing with a bad truth.

`checks` records the arithmetic the bill itself asserts. These are what the parser's
referee uses, and they are verified here by hand so a failing check means the parse is
wrong, not the fixture.

## Checkout captures are bills

Two of the ten were photographed at the checkout screen, before paying,
rather than after. That is deliberate: the receipt a shop app issues once
payment goes through usually drops the line items, so the cart is the only
place the item detail exists. They are marked `captured_before_payment` and
parsed like any other bill.

The one thing the flag is good for is matching: a tip or a coupon applied at
the very last step can make the amount actually paid differ from the total on
the screen, so an exact-amount match is slightly less certain for these.
