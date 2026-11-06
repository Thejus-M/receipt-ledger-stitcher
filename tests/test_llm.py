"""The local-LLM path (plan Stage C, methods C4/C5): a text model cleaning
up OCR output, and a vision model reading the photo directly.

No network and no model here, on purpose: `generate_json` is mocked
throughout, exactly as the OCR engines are never a test dependency. What is
tested is the logic around the model call — turning its JSON into the same
candidate shape the deterministic referee already judges, surviving a reply
that isn't clean JSON, and the one thing that must never regress: the
client only ever talks to this machine.

    .venv/bin/python -m unittest discover -s tests -t .
"""
import json
import unittest
import unittest.mock

from src import llm
from src.parse import LLM_METHODS, _llm_candidates, parse_page_llm
from src.validate import referee


class StaysLocal(unittest.TestCase):
    """The one hard rule. If this ever fails, something tried to make the
    endpoint configurable, which is exactly what must not happen."""

    def test_the_url_is_a_literal_loopback_address(self):
        self.assertTrue(llm.OLLAMA_URL.startswith("http://127.0.0.1"))

    def test_the_url_is_not_read_from_the_environment(self):
        import inspect
        source = inspect.getsource(llm)
        # OLLAMA_URL must not be built from os.environ / os.getenv anywhere.
        self.assertNotIn("environ", source.split("OLLAMA_URL", 1)[1].split("\n")[0])

    def test_every_request_target_is_built_from_the_same_constant(self):
        with unittest.mock.patch("src.llm.urllib.request.urlopen") as mocked:
            mocked.side_effect = OSError("no connection")
            llm.available()
            llm.has_model("qwen2.5:3b")
        for call in mocked.call_args_list:
            arg = call.args[0]
            url = arg.full_url if hasattr(arg, "full_url") else arg
            self.assertTrue(str(url).startswith(llm.OLLAMA_URL))


class JsonExtraction(unittest.TestCase):
    """Models rarely obey 'reply with only JSON' to the letter."""

    def test_plain_json_is_parsed(self):
        self.assertEqual(llm.extract_json('{"total": 456.0}'), {"total": 456.0})

    def test_a_markdown_fence_is_unwrapped(self):
        text = '```json\n{"total": 456.0, "merchant": "KUNNIL"}\n```'
        self.assertEqual(llm.extract_json(text), {"total": 456.0, "merchant": "KUNNIL"})

    def test_a_sentence_of_preamble_is_skipped(self):
        text = 'Here is the extracted data:\n{"total": 120.0}\nLet me know if you need more.'
        self.assertEqual(llm.extract_json(text), {"total": 120.0})

    def test_unparseable_text_is_empty_not_an_exception(self):
        self.assertEqual(llm.extract_json("I could not read this bill clearly."), {})

    def test_empty_input_is_empty(self):
        self.assertEqual(llm.extract_json(""), {})

    def test_a_json_array_is_not_mistaken_for_the_object(self):
        # The schema asks for an object; a bare array reply is not one.
        self.assertEqual(llm.extract_json("[1, 2, 3]"), {})


class PromptBuilding(unittest.TestCase):
    """The schema example embedded in the prompt is full of its own { }
    braces. Building the prompt with str.format() tried to read every one
    of those as a substitution and raised KeyError before a single request
    went out - caught by a live smoke test, not by any mock, which is
    exactly the gap a mocked-only suite leaves: these two tests pin the
    fix without needing a model."""

    def test_the_schema_braces_do_not_break_prompt_building(self):
        prompt = llm.TEXT_PROMPT.replace("<<<OCR_TEXT>>>", "Grand Total : 456.00")
        self.assertIn("Grand Total : 456.00", prompt)
        self.assertIn('"total"', prompt)

    def test_the_ocr_text_is_substituted_not_the_schema(self):
        with unittest.mock.patch("src.llm.generate_json", return_value={}) as mocked:
            llm.structure_from_text("UNIQUE_MARKER_TEXT", model="qwen2.5:3b")
        prompt = mocked.call_args.args[1]
        self.assertIn("UNIQUE_MARKER_TEXT", prompt)
        self.assertNotIn("<<<OCR_TEXT>>>", prompt)


class Unavailable(unittest.TestCase):
    """Ollama not running, or the model not pulled: never a crash."""

    def test_no_connection_gives_an_empty_reply(self):
        with unittest.mock.patch("src.llm.urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertEqual(llm.generate_json("qwen2.5:3b", "prompt"), {})

    def test_availability_check_fails_closed(self):
        with unittest.mock.patch("src.llm.urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertFalse(llm.available())

    def test_a_malformed_response_body_gives_an_empty_reply(self):
        response = unittest.mock.MagicMock()
        response.read.return_value = b"not json at all"
        response.__enter__.return_value = response
        with unittest.mock.patch("src.llm.urllib.request.urlopen", return_value=response):
            self.assertEqual(llm.generate_json("qwen2.5:3b", "prompt"), {})


class Streaming(unittest.TestCase):
    """on_chunk switches to ollama's NDJSON streaming reply - the only way
    to show anything while a vision model is still generating, since the
    plain path otherwise blocks silently for a minute or more."""

    def _ndjson_response(self, *lines: bytes):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.__iter__.return_value = iter(lines)
        return response

    def test_each_fragment_reaches_the_callback_as_it_arrives(self):
        response = self._ndjson_response(
            b'{"response": "{\\"total\\"", "done": false}\n',
            b'{"response": ": 456.0}", "done": true}\n')
        chunks = []
        with unittest.mock.patch("src.llm.urllib.request.urlopen", return_value=response):
            result = llm.generate_json("gemma3n:e2b", "prompt", on_chunk=chunks.append)
        self.assertEqual(chunks, ['{"total"', ': 456.0}'])
        self.assertEqual(result, {"total": 456.0})

    def test_streaming_is_only_requested_when_on_chunk_is_given(self):
        response = self._ndjson_response(b'{"response": "{}", "done": true}\n')
        response.read.return_value = b'{"response": "{}"}'
        with unittest.mock.patch("src.llm.urllib.request.urlopen",
                                 return_value=response) as mocked:
            llm.generate_json("gemma3n:e2b", "prompt", on_chunk=lambda piece: None)
            sent = json.loads(mocked.call_args.args[0].data)
            self.assertTrue(sent["stream"])
            llm.generate_json("gemma3n:e2b", "prompt")
            sent = json.loads(mocked.call_args.args[0].data)
            self.assertFalse(sent["stream"])

    def test_a_dropped_connection_mid_stream_gives_an_empty_reply(self):
        with unittest.mock.patch("src.llm.urllib.request.urlopen", side_effect=OSError("reset")):
            chunks = []
            self.assertEqual(llm.generate_json("gemma3n:e2b", "prompt", on_chunk=chunks.append),
                             {})
            self.assertEqual(chunks, [])


class CandidateConversion(unittest.TestCase):
    """The model's JSON, in the shape the deterministic referee already
    judges - so an LLM reading is scored by the same arithmetic as every
    other method, not by a separate and uncomparable rule."""

    def test_numbers_become_single_item_candidate_lists(self):
        candidates, _ = _llm_candidates({"total": 456.0, "subtotal": 460.82})
        self.assertEqual(candidates["total"], [456.0])
        self.assertEqual(candidates["subtotal"], [460.82])

    def test_a_missing_field_is_an_empty_list_not_a_crash(self):
        candidates, items = _llm_candidates({})
        self.assertEqual(candidates["total"], [])
        self.assertEqual(items, {})

    def test_a_string_amount_is_rejected_rather_than_coerced(self):
        # The schema asks for a plain number; a model that replies "₹456.00"
        # anyway must not have that string silently parsed as if it were
        # trustworthy - it is dropped, and the referee is left with nothing
        # for that field rather than a possibly-wrong guess.
        candidates, _ = _llm_candidates({"total": "₹456.00"})
        self.assertEqual(candidates["total"], [])

    def test_items_carry_through_as_rows_the_referee_can_sum(self):
        _, items = _llm_candidates({"items": [{"description": "TOBLERONE", "amount": 240.16},
                                               {"description": "EGG", "amount": 61.50}]})
        rows = items["llm"][0]
        self.assertEqual([r["amount_candidates"] for r in rows], [[240.16], [61.50]])

    def test_an_item_with_no_readable_amount_is_dropped(self):
        _, items = _llm_candidates({"items": [{"description": "mystery item"}]})
        self.assertEqual(items, {})

    def test_the_first_tax_slabs_cgst_and_sgst_are_read(self):
        candidates, _ = _llm_candidates(
            {"taxes": [{"rate": 5.0, "taxable": 340.47, "cgst": 8.52, "sgst": 8.52}]})
        self.assertEqual((candidates["cgst"], candidates["sgst"]), ([8.52], [8.52]))

    def test_taxable_fills_the_subtotal_when_none_was_given_directly(self):
        candidates, _ = _llm_candidates({"taxes": [{"taxable": 715.10}]})
        self.assertEqual(candidates["subtotal"], [715.10])

    def test_the_converted_candidates_still_balance_through_the_real_referee(self):
        # End to end, without ollama: the KSBC bill's figures, exactly as a
        # model would be expected to return them, scored by the same
        # arithmetic every OCR method goes through.
        candidates, items = _llm_candidates({
            "total": 2550.0,
            "taxes": [{"rate": 251.0, "taxable": 715.10, "cgst": 1794.90, "cess": 40.00}],
            "items": [{"description": "RUM 1000ML", "amount": 1680.0},
                      {"description": "RUM 500ML", "amount": 440.0},
                      {"description": "VODKA 500ML", "amount": 430.0}],
        })
        parse = referee(candidates, items)
        self.assertEqual(parse.total, 2550.0)
        self.assertIn("taxable+tax=total", parse.checks_passed)


class VisionAndTextDispatch(unittest.TestCase):
    """parse_page_llm: the two kinds in LLM_METHODS, both mocked at the
    network boundary."""

    def test_the_bake_off_methods_are_registered(self):
        self.assertIn("F-text-llm", LLM_METHODS)
        self.assertIn("G-vision-llm", LLM_METHODS)
        self.assertEqual(LLM_METHODS["G-vision-llm"][1], "granite3.2-vision:2b")

    def test_a_text_method_reconciles_the_given_ocr_text(self):
        page = unittest.mock.MagicMock(rotation=0, color=None)
        with unittest.mock.patch("src.llm.structure_from_text",
                                 return_value={"total": 456.0, "merchant": "kunnil"}) as mocked:
            parse, _ = parse_page_llm(page, "text", "qwen2.5:3b", quick_text="some ocr text")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], "some ocr text")
        self.assertEqual(parse.total, 456.0)
        self.assertEqual(parse.merchant, "KUNNIL")

    def test_a_vision_method_is_given_the_colour_page(self):
        import numpy as np
        page = unittest.mock.MagicMock(rotation=90, color=np.zeros((10, 10, 3), dtype="uint8"))
        with unittest.mock.patch("src.llm.structure_from_image",
                                 return_value={"total": 250.0}) as mocked:
            parse, _ = parse_page_llm(page, "vision", "granite3.2-vision:2b", quick_text="")
        self.assertIs(mocked.call_args.args[0], page.color)
        self.assertEqual(parse.total, 250.0)
        self.assertEqual(parse.rotation, 90)

    def test_an_unreachable_model_is_a_warning_not_a_crash(self):
        page = unittest.mock.MagicMock(rotation=0, color=None)
        with unittest.mock.patch("src.llm.structure_from_text", return_value={}):
            parse, _ = parse_page_llm(page, "text", "qwen2.5:3b", quick_text="text")
        self.assertIsNone(parse.total)
        self.assertTrue(any("ollama" in w for w in parse.warnings))

    def test_a_cart_screenshot_is_still_read_as_a_bill(self):
        # The LLM is asked to flag a checkout cart, not to reject it - the
        # same rule the deterministic path follows.
        page = unittest.mock.MagicMock(rotation=0, color=None)
        with unittest.mock.patch("src.llm.structure_from_text",
                                 return_value={"total": 139.0, "is_cart": True}):
            parse, _ = parse_page_llm(page, "text", "qwen2.5:3b", quick_text="")
        self.assertTrue(parse.is_cart)
        self.assertEqual(parse.total, 139.0)


if __name__ == "__main__":
    unittest.main()
