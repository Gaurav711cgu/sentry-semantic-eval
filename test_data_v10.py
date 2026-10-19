"""AI-authored split separation and raw candidate labels; no model execution."""

import importlib
import itertools
import json
import unittest
from collections import Counter, defaultdict

from pair_v10 import pair_inputs


class DataTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("data_v10")
        except ImportError:
            self.fail("prospective supervised split dataset missing")

    def test_development_and_final_have_exact_challenge_coverage(self):
        m = self.module()
        for split, count, per_kind in [("dev", 144, 2), ("final", 216, 3)]:
            cases = m.build_split(split)
            self.assertEqual(len(cases), count)
            self.assertEqual(len({c["id"] for c in cases}), count)
            groups = defaultdict(list)
            for case in cases:
                self.assertEqual(
                    set(case),
                    {"id", "kind", "group_id", "request", "orders", "expected_choice"},
                )
                groups[case["group_id"]].append(case)
            self.assertEqual(
                Counter(rows[0]["kind"] for rows in groups.values()),
                {kind: per_kind for kind in m.KINDS},
            )
            for rows in groups.values():
                self.assertEqual(len(rows), 6)
                self.assertEqual(len({c["request"] for c in rows}), 1)
                self.assertEqual(len({c["expected_choice"] for c in rows}), 1)
                self.assertEqual(
                    len({tuple(o["id"] for o in c["orders"]) for c in rows}), 6
                )
                self.assertTrue(
                    all(set(o) == {"id", "status"} for c in rows for o in c["orders"])
                )

    def test_training_has_unique_pairs_and_enough_positive_examples(self):
        m = self.module()
        pairs = m.training_pairs()
        self.assertGreaterEqual(len(pairs), 200)
        self.assertGreaterEqual(sum(row["label"] == 0 for row in pairs), 60)
        self.assertEqual(len({row["prefix_content"] for row in pairs}), len(pairs))
        self.assertTrue(all(set(row) == {"prefix_content", "label"} for row in pairs))
        self.assertTrue(
            all(type(row["label"]) is int and row["label"] in {0, 1} for row in pairs)
        )
        labels = {}
        for case in m.build_split("train"):
            for row in pair_inputs(case["request"], [o["id"] for o in case["orders"]]):
                label = int(row["candidate"] != case["expected_choice"])
                self.assertEqual(labels.setdefault(row["prefix_content"], label), label)
        self.assertEqual(labels, {r["prefix_content"]: r["label"] for r in pairs})

    def test_no_normalized_content_or_template_family_overlap_between_splits(self):
        m = self.module()
        prefixes = {}
        group_ids = {}
        for split in ["train", "dev", "final"]:
            cases = m.build_split(split)
            prefixes[split] = {
                r["prefix_content"]
                for c in cases
                for r in pair_inputs(c["request"], [o["id"] for o in c["orders"]])
            }
            group_ids[split] = {c["group_id"] for c in cases}
        for a, b in itertools.combinations(prefixes, 2):
            self.assertFalse(prefixes[a] & prefixes[b], f"canonical leakage {a}/{b}")
            self.assertFalse(group_ids[a] & group_ids[b])

    def test_absent_targets_and_unavailable_orders_remain_raw_targets(self):
        m = self.module()
        for split in ["train", "dev", "final"]:
            cases = m.build_split(split)
            absent = [c for c in cases if c["kind"] == "absent"]
            self.assertTrue(absent)
            self.assertTrue(
                all(
                    c["expected_choice"] not in {o["id"] for o in c["orders"]}
                    for c in absent
                )
            )
            unavailable = [
                c
                for c in cases
                if c["expected_choice"]
                and any(
                    o["id"] == c["expected_choice"] and o["status"] != "processing"
                    for o in c["orders"]
                )
            ]
            if split == "train":
                self.assertTrue(
                    unavailable, "training must preserve unavailable raw targets"
                )
            else:
                self.assertFalse(
                    unavailable, "evaluation listed positive targets remain eligible"
                )
            for case in absent:
                rows = pair_inputs(case["request"], [o["id"] for o in case["orders"]])
                self.assertIn(case["expected_choice"], [r["candidate"] for r in rows])

    def test_manifest_is_deterministic_and_discloses_ai_authorship(self):
        m = self.module()
        a = m.build_manifest()
        self.assertEqual(a, m.build_manifest())
        self.assertEqual(a["schema"], "candidate-pair-data-v10")
        self.assertTrue(a["references"]["AI_authored"])
        self.assertFalse(a["references"]["independent_human_review"])
        self.assertFalse(a["references"]["real_customers"])
        self.assertFalse(a["independent_final_qualified"])
        self.assertEqual(a["splits"]["dev"]["views"], 144)
        self.assertEqual(a["splits"]["final"]["views"], 216)
        json.dumps(a, allow_nan=False)
        with self.assertRaises(ValueError):
            m.build_split("unknown")


if __name__ == "__main__":
    unittest.main()
