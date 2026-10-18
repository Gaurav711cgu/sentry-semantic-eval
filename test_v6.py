"""BF16 diagnostics cannot hide loaded precision or non-option token output."""

import importlib
import unittest
from copy import deepcopy

import test_contracts as fixtures


class BF16Transcript(unittest.TestCase):
    def test_precision_and_full_vocabulary_diagnostics_are_required(self):
        try:
            c = importlib.import_module("contracts_v6")
        except ImportError:
            self.fail("BF16 transcript validation missing")
        fixture = fixtures.Contracts()
        protocol = fixture.protocol()
        events = fixture.events()
        events[0].update(dtype="bfloat16", load_seconds=2.0, peak_rss_bytes=1024)
        protocol["max_load_seconds"] = 180
        for row in events:
            if row["event"] == "completed":
                row.update(
                    top_token_id=33, top_token_text="B", top_token_is_option=True
                )
        self.assertEqual(len(c.validate_events(events, protocol)), 2)
        for mutate in [
            lambda x: x[0].update(dtype="float32"),
            lambda x: x[0].update(load_seconds=181),
            lambda x: x[0].update(load_seconds=float("nan")),
            lambda x: x[0].update(peak_rss_bytes=protocol["max_rss_bytes"] + 1),
            lambda x: x[2].pop("top_token_id"),
            lambda x: x[2].update(top_token_is_option="yes"),
            lambda x: x[2].update(top_token_text="x" * 129),
        ]:
            changed = deepcopy(events)
            mutate(changed)
            with self.assertRaises(ValueError):
                c.validate_events(changed, protocol)
