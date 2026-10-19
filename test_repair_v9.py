"""Repair experiment contracts; no model imports or scoring."""

import json
import unittest
from pathlib import Path

from repair_v9 import PROMPT, build_protocol, validate_reservation

ROOT = Path(__file__).resolve().parent


class RepairTests(unittest.TestCase):
    def test_one_frozen_candidate_uses_exact_unscored_views(self):
        p = build_protocol(ROOT)
        old = json.loads((ROOT / "protocol_v8_confirm-a.json").read_text())
        self.assertEqual(p["calls"], 156)
        self.assertEqual(p["max_forward_seconds"], 10)
        self.assertEqual(len(p["requests"]), 156)
        self.assertEqual(len(p["logical_requests"]), 288)
        self.assertEqual(p["gold_file"], old["gold_file"])
        self.assertEqual(
            [r["options"] for r in p["logical_requests"]],
            [r["options"] for r in old["logical_requests"]],
        )
        for new, before in zip(p["logical_requests"], old["logical_requests"]):
            self.assertEqual(new["messages"][1], before["messages"][1])
            self.assertEqual(new["messages"][0]["content"], PROMPT)
        self.assertFalse(p["independent_final_qualified"])
        self.assertFalse(p["v8_promotion_eligible"])

    def reservation(self):
        p = build_protocol(ROOT)
        import hashlib

        ledger = {
            "campaign_id": "repair-v9-20261019",
            "budget_max": 296,
            "prior_v8_ledger_sha256": hashlib.sha256(
                (ROOT / "campaign-v8-ledger.json").read_bytes()
            ).hexdigest(),
            "prior_calls_reserved": 140,
            "grant": {"calls": 156},
        }
        return ledger, p

    def test_budget_includes_all_prior_attempts(self):
        ledger, p = self.reservation()
        self.assertEqual(validate_reservation(ledger, ledger["grant"], p, ROOT), 296)

    def test_wrong_prior_identity_is_rejected(self):
        ledger, p = self.reservation()
        ledger["prior_v8_ledger_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_reservation(ledger, ledger["grant"], p, ROOT)

    def test_changed_ceiling_or_calls_is_rejected(self):
        for key, value in [("budget_max", 297), ("prior_calls_reserved", 0)]:
            ledger, p = self.reservation()
            ledger[key] = value
            with self.assertRaises(ValueError):
                validate_reservation(ledger, ledger["grant"], p, ROOT)
        ledger, p = self.reservation()
        p["calls"] = 157
        with self.assertRaises(ValueError):
            validate_reservation(ledger, ledger["grant"], p, ROOT)

    def test_wrong_grant_is_rejected(self):
        ledger, p = self.reservation()
        with self.assertRaises(ValueError):
            validate_reservation(ledger, {"calls": 155}, p, ROOT)

    def test_prompt_covers_failed_mechanisms_without_answer_repair(self):
        for term in [
            "absent",
            "conditional",
            "quoted",
            "questions",
            "revoked",
            "multiple",
            "overrides",
        ]:
            self.assertIn(term, PROMPT)
        self.assertLess(len(PROMPT), 800)


if __name__ == "__main__":
    unittest.main()
