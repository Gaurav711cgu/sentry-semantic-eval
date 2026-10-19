"""Measured-release campaign contracts; no weights or inference."""

import importlib
import json
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path


class CampaignContracts(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("campaign_v8")
        except ImportError:
            self.fail("bounded compact-prompt campaign missing")

    def test_exact_calibration_dedup_preserves_every_raw_request(self):
        m = self.module()
        old = json.loads(Path("protocol_v6.json").read_text())
        logical = m.requests(old["requests"], "a")
        physical, bindings = m.deduplicate(logical, old["model"], old["runtime"])
        self.assertEqual(len(logical), 108)
        self.assertEqual(len(physical), 66)
        self.assertEqual(
            Counter(r["mode"] for r in physical), {"raw": 54, "content_free": 12}
        )
        self.assertEqual(len(bindings), 108)
        lookup = {(r["id"], r["mode"]): r for r in physical}
        for r, b in zip(logical, bindings):
            actual = lookup[(b["physical_id"], b["mode"])]
            self.assertEqual(r["messages"], actual["messages"])
            self.assertEqual(r["options"], actual["options"])

    def test_cache_identity_changes_with_model_runtime_options_or_prompt(self):
        m = self.module()
        old = json.loads(Path("protocol_v6.json").read_text())
        r = m.requests(old["requests"], "a")[1]
        key = m.cache_key(r, old["model"], old["runtime"])
        for row, model, runtime in [
            (
                {**r, "options": list(reversed(r["options"]))},
                old["model"],
                old["runtime"],
            ),
            (
                {
                    **r,
                    "messages": [
                        {"role": "system", "content": "other"},
                        r["messages"][1],
                    ],
                },
                old["model"],
                old["runtime"],
            ),
            (r, {**old["model"], "revision": "changed"}, old["runtime"]),
            (r, old["model"], {**old["runtime"], "python": "changed"}),
        ]:
            self.assertNotEqual(key, m.cache_key(row, model, runtime))

    def test_confirmatory_set_has_fixed_coverage_labels_and_permutations(self):
        m = self.module()
        cases = m.confirmatory_cases()
        self.assertEqual(len(cases), 144)
        self.assertEqual(len({c["family"] for c in cases}), 24)
        self.assertEqual(len({c["kind"] for c in cases}), 12)
        self.assertEqual(Counter(c["kind"] for c in cases).most_common()[0][1], 12)
        self.assertEqual(sum(c["effect_target"] is not None for c in cases), 24)
        for family in {c["family"] for c in cases}:
            group = [c for c in cases if c["family"] == family]
            self.assertEqual(
                len({tuple(o["order_id"] for o in c["orders"]) for c in group}), 6
            )
            self.assertEqual(len({c["request"] for c in group}), 1)
            self.assertEqual(
                len({(c["raw_target"], c["effect_target"]) for c in group}), 1
            )
        logical = m.confirmatory_requests(cases, "b")
        physical, bindings = m.deduplicate(
            logical, {"model": "frozen"}, {"runtime": "frozen"}
        )
        self.assertEqual(len(physical), 156)
        self.assertEqual(len(bindings), 288)

    def test_transcript_expansion_rejects_missing_duplicate_or_wrong_cache_identity(
        self,
    ):
        m = self.module()
        old = json.loads(Path("protocol_v6.json").read_text())
        logical = m.requests(old["requests"], "a")
        physical, bindings = m.deduplicate(logical, old["model"], old["runtime"])
        p = {
            "requests": physical,
            "bindings": bindings,
            "logical_requests": logical,
            "model": old["model"],
            "runtime": old["runtime"],
        }
        rows = [
            {
                "id": r["id"],
                "mode": r["mode"],
                "options": r["options"],
                "logits": list(range(len(r["options"]))),
            }
            for r in physical
        ]
        self.assertEqual(len(m.expand(rows, p)), 108)
        for changed in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                m.expand(changed, p)
        broken = deepcopy(p)
        broken["bindings"][0]["cache_key"] = "wrong"
        with self.assertRaises(ValueError):
            m.expand(rows, broken)

    def test_ledger_rejects_reused_runs_overbudget_and_two_confirmatory_attempts(self):
        m = self.module()
        self.assertTrue(
            hasattr(m, "validate_ledger"), "whole campaign budget guard missing"
        )
        ledger = {
            "campaign_id": "measured-v8-20261019",
            "budget_max": 296,
            "grants": {"profile": {"calls": 8, "run_id": 1}},
        }
        self.assertEqual(m.validate_ledger(ledger), 8)
        for grants in [
            {"profile": {"calls": 9, "run_id": 1}},
            {"profile": {"calls": 8, "run_id": 1}, "dev-a": {"calls": 66, "run_id": 1}},
            {
                "confirm-a": {"calls": 156, "run_id": 1},
                "confirm-b": {"calls": 156, "run_id": 2},
            },
        ]:
            with self.assertRaises(ValueError):
                m.validate_ledger({**ledger, "grants": grants})

    def test_selection_requires_passing_execution_performance_and_semantics(self):
        m = self.module()
        self.assertTrue(
            hasattr(m, "select_candidate"), "deterministic promotion selection missing"
        )
        passing = {
            "execution_status": "completed",
            "performance_status": "passed",
            "evidence_status": "verified",
            "semantic_status": "passed",
            "raw_summary": {"raw_accuracy": 1.0, "eligible_recall": 1.0},
            "max_forward_seconds": 3.0,
        }
        self.assertEqual(m.select_candidate({"a": passing, "b": passing}), "a")
        self.assertEqual(
            m.select_candidate(
                {"a": {**passing, "performance_status": "failed"}, "b": passing}
            ),
            "b",
        )
        self.assertIsNone(
            m.select_candidate({"a": {**passing, "semantic_status": "failed"}})
        )

    def test_completion_never_confuses_execution_quality_or_cleanup(self):
        m = self.module()
        self.assertTrue(
            hasattr(m, "finish"), "separate stage completion contract missing"
        )
        kwargs = dict(
            complete=True, graded=True, stopped=True, removed=True, failure=None
        )
        self.assertTrue(m.finish({"stage_gate_passed": True}, **kwargs)["passed"])
        failed = m.finish({"stage_gate_passed": False}, **kwargs)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["execution_status"], "completed")
        self.assertEqual(failed["evidence_status"], "verified")
        self.assertFalse(
            m.finish({"stage_gate_passed": True}, **{**kwargs, "removed": False})[
                "passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
