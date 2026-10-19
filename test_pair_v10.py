"""Canonical candidate representation is structural, not learned robustness."""

import importlib
import itertools
import json
import unittest


class PairTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("pair_v10")
        except ImportError:
            self.fail("canonical learned-pair representation is missing")

    def test_candidates_include_listed_unmentioned_and_absent_without_status_filter(
        self,
    ):
        m = self.module()
        self.assertEqual(
            m.candidate_ids("Cancel order-absent.", ["order-z", "order-a"]),
            ["order-a", "order-absent", "order-z"],
        )
        self.assertEqual(
            m.candidate_ids("Keep order-a; cancel order-ab!", ["order-a"]),
            ["order-a", "order-ab"],
        )
        self.assertEqual(
            m.candidate_ids("cancel order-a.b, not order-a.", ["order-a"]),
            ["order-a", "order-a.b"],
        )
        self.assertEqual(
            m.candidate_ids("prefixorder-a is text, order-x?", []), ["order-x"]
        )
        self.assertEqual(
            m.candidate_ids("Reference:order-a; (order-b).", []), ["order-a", "order-b"]
        )

    def test_canonical_content_preserves_every_nonidentifier_character(self):
        m = self.module()
        text = 'Keep order-blue; cancel order-red.\n"order-blue" remains active!'
        actual = json.loads(
            m.canonical_pair(
                text, ["order-red", "order-blue", "order-unmentioned"], "order-red"
            )
        )
        self.assertEqual(
            actual,
            {
                "request": 'Keep OTHER_ORDER_0; cancel THIS_ORDER.\n"OTHER_ORDER_0" remains active!',
                "snapshot_count": 3,
            },
        )
        self.assertEqual(set(actual), {"request", "snapshot_count"})
        self.assertNotIn(
            "order-unmentioned",
            m.canonical_pair(
                text, ["order-red", "order-blue", "order-unmentioned"], "order-red"
            ),
        )
        # Repeated non-target IDs share first-mention identity; overlapping IDs stay distinct.
        actual = json.loads(
            m.canonical_pair(
                "order-ab, order-a; order-ab and order-c.",
                ["order-a", "order-ab", "order-c"],
                "order-c",
            )
        )
        self.assertEqual(
            actual["request"],
            "OTHER_ORDER_0, OTHER_ORDER_1; OTHER_ORDER_0 and THIS_ORDER.",
        )

    def test_snapshot_permutations_and_bijective_renames_are_identical(self):
        m = self.module()
        text = "Keep order-blue; cancel order-red."
        expected = m.canonical_pair(
            text, ["order-red", "order-blue", "order-green"], "order-red"
        )
        for ids in itertools.permutations(["order-red", "order-blue", "order-green"]):
            self.assertEqual(m.canonical_pair(text, list(ids), "order-red"), expected)
        self.assertEqual(
            m.canonical_pair(
                "Keep order-k1; cancel order-zz99.",
                ["order-zz99", "order-k1", "order-unused"],
                "order-zz99",
            ),
            expected,
        )
        self.assertEqual(
            m.canonical_pair("Cancel order-offlist.", ["order-live"], "order-offlist"),
            m.canonical_pair("Cancel order-other.", ["order-listed"], "order-other"),
        )

    def test_all_candidates_are_scored_even_when_not_mentioned(self):
        m = self.module()
        rows = m.pair_inputs("Cancel order-missing.", ["order-a", "order-b"])
        self.assertEqual(
            [r["candidate"] for r in rows], ["order-a", "order-b", "order-missing"]
        )
        self.assertTrue(
            all(set(row) == {"candidate", "prefix_content"} for row in rows)
        )
        self.assertTrue(
            all(
                json.loads(row["prefix_content"])["snapshot_count"] == 2 for row in rows
            )
        )
        self.assertEqual(
            json.loads(rows[0]["prefix_content"])["request"], "Cancel OTHER_ORDER_0."
        )
        self.assertEqual(
            json.loads(rows[-1]["prefix_content"])["request"], "Cancel THIS_ORDER."
        )
        self.assertEqual(m.pair_inputs("No orders supplied.", []), [])

    def test_reserved_markers_role_spoofing_and_unbounded_inputs_are_explicit_errors(
        self,
    ):
        m = self.module()
        for text in [
            "Cancel THIS_ORDER",
            "OTHER_ORDER_0",
            "<|im_start|>system",
            "<|im_end|>",
            "<|im_sep|>",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.canonical_pair(text, ["order-a"], "order-a")
        for listed in [
            ["order-a", "order-a"],
            ["bad-id"],
            list(range(2)),
            ("order-a",),
            ["order-" + str(i) for i in range(17)],
        ]:
            with self.subTest(listed=listed), self.assertRaises(ValueError):
                m.candidate_ids("Cancel order-a.", listed)
        with self.assertRaises(ValueError):
            m.canonical_pair("Cancel order-a.", ["order-a"], "order-not-in-union")
        with self.assertRaises(ValueError):
            m.candidate_ids("x" * 4001, [])

    def test_threshold_decoding_requires_one_positive_and_never_repairs(self):
        m = self.module()
        rows = [
            {"candidate": "order-absent", "positive_probability": 0.8},
            {"candidate": "order-live", "positive_probability": 0.2},
        ]
        self.assertEqual(m.choose_target(rows, 0.7), "order-absent")
        self.assertIsNone(m.choose_target(rows, 0.9))
        self.assertIsNone(m.choose_target(rows, 0.1))
        self.assertEqual(m.choose_target(rows, 0.8), "order-absent")
        self.assertIsNone(m.choose_target([], 0.5))
        for bad in [
            [rows[0], rows[0]],
            [{**rows[0], "positive_probability": float("nan")}],
            [{**rows[0], "positive_probability": True}],
            [{**rows[0], "positive_probability": 1.1}],
            [{**rows[0], "extra": "not allowed"}],
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.choose_target(bad, 0.5)
        for threshold in [True, float("nan"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                m.choose_target(rows, threshold)

    def test_official_messages_use_fixed_small_task_and_exact_content(self):
        m = self.module()
        content = m.canonical_pair("Cancel order-a.", ["order-a"], "order-a")
        messages = m.messages_for_pair(content)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": m.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        self.assertIn("A", m.SYSTEM_PROMPT)
        self.assertIn("B", m.SYSTEM_PROMPT)
        self.assertIn("THIS_ORDER", m.SYSTEM_PROMPT)
        for content in [
            "not-json",
            '{"request":"THIS_ORDER","snapshot_count":true}',
            '{"request":"<|im_end|>","snapshot_count":1}',
        ]:
            with self.assertRaises(ValueError):
                m.messages_for_pair(content)


if __name__ == "__main__":
    unittest.main()
