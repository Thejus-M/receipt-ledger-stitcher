"""The decisions this project makes, tested on the shapes the real bills had.

No OCR and no browser: each test is text or numbers in, a decision out. The
cases are drawn from the ten photographed bills, and several of them exist
because the first implementation got them wrong.

    .venv/bin/python -m unittest discover -s tests -t .
"""
import unittest

from src import fields as F
from src import match as M
from src.parse import _llm_candidates, reconcile_date
from src.validate import Parse, add_warnings, choose_item_amounts, referee, score_confidence


class MoneyReadings(unittest.TestCase):
    def test_a_bare_integer_offers_its_decimal_readings(self):
        # Thermal print loses the point: "461" on the bill is 4.61.
        self.assertIn(4.61, F.money_candidates("461"))
        self.assertIn(461.0, F.money_candidates("461"))

    def test_a_printed_decimal_is_taken_at_face_value_first(self):
        self.assertEqual(F.money_candidates("456.00")[0], 456.0)

    def test_the_rupee_mark_and_commas_are_not_part_of_the_number(self):
        self.assertEqual(F.money_candidates("₹ 1,18,272.23")[0], 118272.23)

    def test_a_year_is_not_money(self):
        self.assertIsNone(F.MONEY_RE.fullmatch("2026"))
        self.assertTrue(F.MONEY_RE.fullmatch("₹ 4081.82"))
        self.assertTrue(F.MONEY_RE.fullmatch("- ₹ 89.82"))


class Labels(unittest.TestCase):
    def test_a_label_takes_the_figure_on_its_own_line(self):
        lines = ["Bill Discount : 461", "Grand Total : 456.00"]
        self.assertIn(4.61, F.find_labelled_value(lines, F.DISCOUNT_LABELS))

    def test_a_label_with_no_figure_does_not_borrow_the_next_ones(self):
        # The bug this replaced read the discount as 456.00, the grand total
        # two lines below it.
        lines = ["Bill Discount :", "", "Grand Total : 456.00"]
        self.assertNotIn(456.0, F.find_labelled_value(lines, F.DISCOUNT_LABELS))

    def test_a_label_mid_sentence_is_not_a_label(self):
        lines = ["Looking for a higher credit limit?", "Click here to check"]
        self.assertEqual(F.find_labelled_value(lines, (r"credit limit",)), [])


class PaymentAndKind(unittest.TestCase):
    def test_a_named_rail_beats_the_word_cash(self):
        # Real bills print "GPAY 456.00" and "Tender Cash 456.00" together,
        # and one prints "To: CASH" above "Card : 149.00".
        self.assertEqual(F.find_payment_hint("GPAY : 456.00\nTender Cash : 456.00"), "UPI")
        self.assertEqual(F.find_payment_hint("To : CASH\nCard : 149.00"), "CARD")

    def test_cash_is_believed_when_nothing_else_is_named(self):
        self.assertEqual(F.find_payment_hint("Bill To : CASH"), "CASH")

    def test_a_checkout_capture_is_recognised(self):
        cart = "Clear Cart\nApply Coupon\nPlace Order\nDelivery Tip"
        self.assertTrue(F.looks_like_cart(cart))

    def test_a_paid_bill_is_not_a_cart(self):
        self.assertFalse(F.looks_like_cart("Grand Total 456.00\nPaid via UPI\nThank you"))


class Dates(unittest.TestCase):
    def test_two_digit_years_become_this_century(self):
        self.assertEqual(F.parse_date("04/11/26"), "2026-11-04")

    def test_the_photo_fixes_a_misread_year(self):
        # A bill dated 2026 was OCR'd as 2025; the day and month agree, so
        # the camera's year wins and the match is not lost.
        warnings = []
        self.assertEqual(reconcile_date("2025-11-05", "2026-11-05", warnings), "2026-11-05")
        self.assertTrue(warnings)

    def test_a_genuinely_different_date_is_kept_from_the_bill(self):
        # Photographed days after the purchase: the bill is right.
        warnings = []
        self.assertEqual(reconcile_date("2026-08-31", "2026-09-04", warnings), "2026-08-31")


class Referee(unittest.TestCase):
    def test_the_reading_that_balances_wins(self):
        # The real case: 460.82 - 4.61 - 0.21 = 456.00, while the same line
        # read as 4.81 gives 455.80 and does not balance.
        parse = referee({"total": [456.0], "subtotal": [460.82],
                         "discount": [4.81, 4.61], "round_off": [0.21]}, {})
        self.assertEqual(parse.discount, 4.61)
        self.assertIn("subtotal-discount-roundoff=total", parse.checks_passed)

    def test_tax_identity_is_recognised(self):
        parse = referee({"total": [1592.0], "subtotal": [1516.2],
                         "cgst": [37.9], "sgst": [37.9]}, {})
        self.assertIn("taxable+tax=total", parse.checks_passed)

    def test_a_truncated_figure_is_recovered_from_the_identity(self):
        # The KSBC bill's right edge is cut off: 715.1_, 1794.9_ and 40.0_.
        parse = referee({"total": [2550.0], "subtotal": [715.1, 7151.0],
                         "cgst": [1794.9], "cess": [40.0]}, {})
        self.assertEqual(parse.subtotal, 715.1)

    def test_no_total_is_an_honest_failure(self):
        parse = referee({"total": []}, {})
        self.assertIsNone(parse.total)
        self.assertTrue(parse.warnings)


class ItemAmounts(unittest.TestCase):
    def test_the_combination_that_sums_to_the_subtotal_is_chosen(self):
        chosen = choose_item_amounts([[2401.6, 240.16], [103.31], [723.5, 72.35], [450.0, 45.0]],
                                     target=46082)
        self.assertEqual(chosen, [240.16, 103.31, 72.35, 45.0])

    def test_without_a_target_the_first_reading_is_kept(self):
        self.assertEqual(choose_item_amounts([[12.0, 1.2], [3.0]], None), [12.0, 3.0])

    def test_an_impossible_target_falls_back_rather_than_inventing(self):
        self.assertEqual(choose_item_amounts([[10.0], [20.0]], target=999999), [10.0, 20.0])

    def test_a_cleaner_but_shorter_reading_beats_a_noisier_longer_one(self):
        # A consensus run pools a clean 4-row reading with a noisy 6-row one
        # that split two lines in half. Picking by row count alone would
        # keep the noisy 6; the referee must keep whichever set actually
        # sums to the target.
        clean = [{"description": "A", "amount_candidates": [240.16]},
                 {"description": "B", "amount_candidates": [103.31]},
                 {"description": "C", "amount_candidates": [72.35]},
                 {"description": "D", "amount_candidates": [45.00]}]
        noisy = [{"description": "A", "amount_candidates": [240.16]},
                 {"description": "A-tail", "amount_candidates": [0.0]},
                 {"description": "B", "amount_candidates": [103.31]},
                 {"description": "C", "amount_candidates": [72.35]},
                 {"description": "D", "amount_candidates": [45.00]},
                 {"description": "junk", "amount_candidates": [999.0]}]
        parse = referee({"total": [456.0], "subtotal": [460.82]},
                        {"single": [noisy, clean]})
        self.assertEqual(len(parse.items), 4)
        self.assertEqual(round(sum(i["amount"] for i in parse.items), 2), 460.82)


class Warnings(unittest.TestCase):
    def test_items_that_do_not_add_up_are_called_out(self):
        parse = Parse(total=456.0, subtotal=460.82,
                      items=[{"description": "X", "amount": 100.0}], date="2026-11-05")
        add_warnings(parse)
        self.assertTrue(any("add up to" in w for w in parse.warnings))

    def test_a_checkout_capture_is_noted_but_still_a_bill(self):
        # These are photographed at the checkout on purpose: the receipt
        # issued after payment drops the line items. So it is a bill, and
        # the note is about the amount possibly changing at the last step,
        # not about rejecting it.
        parse = Parse(total=150.0, is_cart=True, date="2026-09-04")
        add_warnings(parse)
        self.assertTrue(any("checkout" in w for w in parse.warnings))
        self.assertFalse(any("not a paid bill" in w for w in parse.warnings))

    def test_a_self_contradicting_gst_breakdown_is_flagged(self):
        # Real gemma3n:e2b reply on the Cake World bill: total/subtotal/items
        # were right, but cgst 5.00 + sgst 9.00 = 14.00 while the bill's own
        # total minus subtotal is really 7.10 — the model's tax split
        # disagrees with its own other numbers.
        data = {
            "merchant": "CAKE WORLD", "total": 149.0, "subtotal": 141.9,
            "discount": 0.0, "round_off": 0.0,
            "items": [{"description": "MILK BROWNIE BOX", "amount": 149.0}],
            "taxes": [{"rate": 14.9, "taxable": 149.0, "cgst": 5.0, "sgst": 9.0}],
        }
        candidates, items = _llm_candidates(data)
        parse = referee(candidates, items)
        parse.confidence = score_confidence(parse, 1.0)
        add_warnings(parse)
        self.assertTrue(parse.needs_review)
        self.assertTrue(any("GST breakdown" in w for w in parse.warnings))

    def test_a_gst_free_bill_is_not_penalised_for_lacking_a_split(self):
        parse = Parse(total=149.0, subtotal=141.9, date="2026-09-04")
        add_warnings(parse)
        self.assertFalse(any("GST breakdown" in w for w in parse.warnings))


class Matching(unittest.TestCase):
    KUNNIL = {"merchant": "KUNNIL MARGINFREE HYPERMARKET", "date": "2026-11-05", "total": 456.0}

    def _txn(self, **kw):
        base = {"id": "t1", "date": "2026-11-05", "merchant": "Upi-kunnil Marginfree",
                "amount": -456.0, "tracked": False}
        base.update(kw)
        return base

    def test_exact_amount_same_day_and_name_scores_high(self):
        points, reasons = M.score(self.KUNNIL, self._txn())
        self.assertGreaterEqual(points, 85)
        self.assertIn("exact amount", reasons)

    def test_money_in_is_never_a_candidate(self):
        self.assertEqual(M.score(self.KUNNIL, self._txn(amount=456.0))[0], 0)

    def test_a_different_merchant_is_penalised_not_merely_unrewarded(self):
        # The real near-miss: a 250 restaurant bill and a 250 PVR Inox
        # ticket three days apart. Amount and date alone rank it as a good
        # match; only the name says otherwise.
        bill = {"merchant": "LOADED RESTAURANT", "date": "2026-09-16", "total": 250.0}
        pvr = self._txn(date="2026-09-19", merchant="Upi-pvr Inox Limited", amount=-250.0)
        points, reasons = M.score(bill, pvr)
        self.assertLess(points, M.AUTO_FLOOR)
        self.assertIn("merchant does not match", reasons)

    def test_a_name_inside_a_longer_one_still_counts(self):
        bill = {"merchant": "K.S.B.C. 6020-CHINNAKANAL", "date": "2026-08-30", "total": 2550.0}
        txn = self._txn(date="2026-08-30", amount=-2550.0,
                        merchant="UPI/P2M/165729881322/KSBC FL01 6020 CHINNA/Liq")
        self.assertGreaterEqual(M.score(bill, txn)[0], 85)

    def test_rails_and_reference_numbers_are_not_merchant_words(self):
        self.assertEqual(M.tokens("UPI/P2M/165729881322/HDFC BANK"), set())

    def test_a_far_away_date_is_disqualified_inside_the_window(self):
        self.assertEqual(M.score(self.KUNNIL, self._txn(date="2026-10-01"), max_days=5)[0], 0)

    def test_the_whole_history_can_be_searched_for_the_report(self):
        points, _ = M.score(self.KUNNIL, self._txn(date="2026-10-01"), max_days=None)
        self.assertGreater(points, 0)

    def test_auto_pick_needs_a_clear_winner(self):
        first = M.Candidate(self._txn(), 95, ["exact amount"])
        close = M.Candidate(self._txn(id="t2"), 90, ["exact amount"])
        self.assertIsNone(M.auto_pick([first, close]))
        self.assertIs(M.auto_pick([first, M.Candidate(self._txn(id="t3"), 60, [])]), first)

    def test_untracked_only_filters_the_candidates(self):
        txns = [self._txn(id="tracked-one", tracked=True)]
        self.assertEqual(M.rank(self.KUNNIL, txns, untracked_only=True), [])
        self.assertEqual(len(M.rank(self.KUNNIL, txns, untracked_only=False)), 1)


if __name__ == "__main__":
    unittest.main()


class TotalsAndRates(unittest.TestCase):
    """Two systematic misreads the first bench exposed."""

    def test_a_sub_total_is_not_the_grand_total(self):
        lines = ["Total Qty: 1  Sub Total 238.10", "CGST 2.5% 5.95", "Grand Total 250.00"]
        self.assertEqual(F.find_total(lines)[0], 250.0)

    def test_an_item_total_is_not_the_grand_total(self):
        lines = ["Item Total Amount : 460.82", "Grand Total : 456.00"]
        self.assertEqual(F.find_total(lines)[0], 456.0)

    def test_a_quantity_is_not_an_amount(self):
        self.assertNotIn(1.0, F.find_total(["Total Qty: 1"]))

    def test_a_gst_rate_is_not_the_gst_amount(self):
        # "CGST@2.5 2.5% 9.05" used to yield 2.5.
        self.assertEqual(F.find_labelled_value(["CGST@2.5 2.5% 9.05"], F.CGST_LABELS)[0], 9.05)

    def test_a_percentage_column_does_not_hide_the_amount(self):
        self.assertEqual(F.find_labelled_value(["SGST 2.5% 5.95"], F.SGST_LABELS)[0], 5.95)
