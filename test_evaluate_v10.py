"""Scoring and development selection are independent of inference and controllers."""

import copy
import importlib
import itertools
import json
import unittest

from pair_v10 import canonical_pair, pair_inputs


def group(name, request, expected, status="processing", *, split="dev"):
    orders = [
        {"id": "order-red", "status": status},
        {"id": "order-blue", "status": "processing"},
        {"id": "order-green", "status": "shipped"},
    ]
    return [
        {
            "id": f"{split}-{name}-{index}",
            "group_id": f"{split}-{name}",
            "kind": name,
            "request": request,
            "orders": list(permutation),
            "expected_choice": expected,
        }
        for index, permutation in enumerate(itertools.permutations(orders))
    ]


def records_for(cases, *, positive=0.85, negative=0.1, duration=0.2):
    records = {}
    for case in cases:
        for row in pair_inputs(case["request"], [o["id"] for o in case["orders"]]):
            records[row["prefix_content"]] = {
                "positive_probability": positive
                if row["candidate"] == case["expected_choice"]
                else negative,
                "duration_seconds": duration,
            }
    return records


class EvaluateTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("evaluate_v10")
        except ImportError:
            self.fail("v10 independent scoring and development selector are missing")

    def test_exact_targets_pair_metrics_and_cache_alias_counts(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        cases += group("negation", "Do not cancel order-red.", None)
        cases += group("absent", "Cancel order-missing.", "order-missing")
        records = records_for(cases)
        report = m.score_cases(cases, records, 0.7)
        self.assertEqual(report["cases"], 18)
        self.assertEqual(report["raw_exact_correct"], 18)
        self.assertEqual(report["raw_accuracy"], 1)
        self.assertEqual(report["eligible_targets"], 6)
        self.assertEqual(report["eligible_raw_target_hits"], 6)
        self.assertEqual(report["eligible_recall"], 1)
        self.assertEqual(report["permutation_groups"], 3)
        self.assertEqual(report["consistent_groups"], 3)
        self.assertEqual(report["exact_groups"], 3)
        self.assertEqual(report["logical_scoring_calls"], 60)
        # Present/absent target status is intentionally absent from LM inputs.
        # Their identical masked requests share the same physical computation.
        self.assertEqual(report["physical_scoring_calls"], 4)
        self.assertEqual(report["cache_aliases"], 56)
        self.assertEqual(
            report["pair_counts"],
            {
                "true_positive": 1,
                "false_positive": 0,
                "true_negative": 3,
                "false_negative": 0,
            },
        )
        self.assertEqual(
            report["logical_pair_counts"],
            {
                "true_positive": 12,
                "false_positive": 0,
                "true_negative": 48,
                "false_negative": 0,
            },
        )
        self.assertEqual(report["pair_balanced_accuracy"], 1)
        self.assertEqual(report["pair_precision"], 1)
        self.assertEqual(report["choices"]["dev-absent-0"], "order-missing")
        self.assertFalse(report["controller_validation_performed"])
        self.assertFalse(report["independent_final_qualified"])
        self.assertEqual(
            report["invariance_scope"],
            "structural canonicalization, not learned permutation robustness",
        )
        self.assertAlmostEqual(report["max_draft_forward_seconds"], 0.8)
        self.assertAlmostEqual(report["logical_forward_seconds"], 12)
        self.assertAlmostEqual(report["physical_forward_seconds"], 0.8)
        self.assertTrue(report["draft_latency_gate_passed"])

    def test_processing_only_eligibility_never_changes_raw_or_pair_labels(self):
        m = self.module()
        for status, eligible in [
            ("processing", 6),
            ("pending", 0),
            ("shipped", 0),
            ("cancelled", 0),
        ]:
            cases = group("status", "Cancel order-red.", "order-red", status)
            report = m.score_cases(cases, records_for(cases), 0.7)
            self.assertEqual(report["raw_accuracy"], 1)
            self.assertEqual(report["eligible_targets"], eligible)
            self.assertEqual(report["eligible_recall"], 1 if eligible else None)
            self.assertEqual(report["pair_counts"]["true_positive"], 1)
        self.assertIn("processing", m.ELIGIBILITY_RULE)
        self.assertNotIn("pending", m.ELIGIBLE_STATUSES)

    def test_pair_balanced_accuracy_is_physical_and_undefined_classes_are_explicit(
        self,
    ):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        records = records_for(cases, positive=0.1, negative=0.9)
        report = m.score_cases(cases, records, 0.7)
        self.assertEqual(report["raw_accuracy"], 0)
        self.assertEqual(report["eligible_recall"], 0)
        self.assertEqual(
            report["pair_counts"],
            {
                "true_positive": 0,
                "false_positive": 1,
                "true_negative": 0,
                "false_negative": 1,
            },
        )
        self.assertEqual(report["pair_balanced_accuracy"], 0)
        self.assertEqual(report["pair_precision"], 0)
        self.assertIsNone(report["choices"][cases[0]["id"]])
        negative = group("negative", "Keep order-red.", None)
        report = m.score_cases(negative, records_for(negative), 0.7)
        self.assertIsNone(report["pair_balanced_accuracy"])
        self.assertIsNone(report["pair_precision"])

    def test_raw_identity_and_unique_record_coverage_are_required(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        records = records_for(cases)
        contents = list(records)
        self.assertEqual(set(m.validate_score_records(contents, records)), set(records))
        list_records = [
            {"prefix_content": content, **record} for content, record in records.items()
        ]
        self.assertEqual(m.validate_score_records(contents, list_records), records)
        invalid = [
            dict(records, extra={"positive_probability": 0.5, "duration_seconds": 0}),
            {content: value for content, value in list(records.items())[1:]},
            list_records + list_records[:1],
        ]
        for probability in [True, float("nan"), float("inf"), -0.1, 1.1, "0.8"]:
            changed = copy.deepcopy(records)
            changed[contents[0]]["positive_probability"] = probability
            invalid.append(changed)
        for duration in [True, float("nan"), float("inf"), -1, "1"]:
            changed = copy.deepcopy(records)
            changed[contents[0]]["duration_seconds"] = duration
            invalid.append(changed)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.validate_score_records(contents, value)
        with self.assertRaises(ValueError):
            m.validate_score_records(contents + contents[:1], records)
        with self.assertRaises(ValueError):
            m.score_cases(cases + cases[:1], records, 0.7)

    def test_permutation_groups_must_preserve_six_unique_snapshots_and_labels(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        invalid_groups = [cases[:-1], cases + [dict(cases[0], id="dev-extra")]]
        changed = copy.deepcopy(cases)
        changed[-1]["orders"] = changed[0]["orders"]
        invalid_groups.append(changed)
        changed = copy.deepcopy(cases)
        changed[-1]["expected_choice"] = None
        invalid_groups.append(changed)
        changed = copy.deepcopy(cases)
        changed[-1]["orders"] = [dict(order) for order in changed[-1]["orders"]]
        changed[-1]["orders"][0]["status"] = "delivered"
        invalid_groups.append(changed)
        for malformed in invalid_groups:
            with self.subTest(cases=malformed), self.assertRaises(ValueError):
                m.score_cases(malformed, records_for(malformed), 0.7)

    def test_case_schema_and_contradictory_canonical_labels_fail_closed(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        opposite = group("opposite", "Cancel order-red.", None)
        with self.assertRaises(ValueError):
            m.score_cases(cases + opposite, records_for(cases + opposite), 0.7)
        for field, value in [
            ("expected_choice", True),
            ("request", ""),
            ("orders", []),
            ("extra", "unknown"),
        ]:
            changed = copy.deepcopy(cases)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.score_cases(changed, records_for(cases), 0.7)

    def test_latency_gate_uses_sum_per_draft_and_missing_timing_is_not_a_pass(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red")
        report = m.score_cases(cases, records_for(cases, duration=4), 0.7)
        self.assertEqual(report["max_pair_forward_seconds"], 4)
        self.assertEqual(report["max_draft_forward_seconds"], 12)
        self.assertFalse(report["draft_latency_gate_passed"])
        scalars = {
            content: row["positive_probability"]
            for content, row in records_for(cases).items()
        }
        report = m.score_cases(cases, scalars, 0.7)
        self.assertIsNone(report["max_draft_forward_seconds"])
        self.assertIsNone(report["draft_latency_gate_passed"])
        self.assertTrue(
            all(row["draft_forward_seconds"] is None for row in report["case_results"])
        )
        self.assertTrue(
            all(row["selected_choice"] == "order-red" for row in report["case_results"])
        )

    def test_cache_identity_binds_model_runtime_adapter_prefix_and_prompt(self):
        m = self.module()
        content = canonical_pair("Cancel order-red.", ["order-red"], "order-red")
        identity = m.cache_identity(
            {"revision": "r1"},
            {"torch": "t1", "device": "cpu"},
            {"sha256": "a1", "step": 10},
            content,
        )
        self.assertEqual(
            identity,
            m.cache_identity(
                {"revision": "r1"},
                {"device": "cpu", "torch": "t1"},
                {"step": 10, "sha256": "a1"},
                content,
            ),
        )
        for model, runtime, adapter, prefix in [
            (
                {"revision": "r2"},
                {"torch": "t1", "device": "cpu"},
                {"sha256": "a1", "step": 10},
                content,
            ),
            (
                {"revision": "r1"},
                {"torch": "t2", "device": "cpu"},
                {"sha256": "a1", "step": 10},
                content,
            ),
            (
                {"revision": "r1"},
                {"torch": "t1", "device": "cpu"},
                {"sha256": "a2", "step": 10},
                content,
            ),
            (
                {"revision": "r1"},
                {"torch": "t1", "device": "cpu"},
                {"sha256": "a1", "step": 10},
                canonical_pair("Keep order-red.", ["order-red"], "order-red"),
            ),
        ]:
            self.assertNotEqual(
                identity, m.cache_identity(model, runtime, adapter, prefix)
            )
        with self.assertRaises(ValueError):
            m.cache_identity(
                {"temperature": float("nan")}, {"device": "cpu"}, None, content
            )

    def test_development_selection_requires_base_and_tie_prefers_lower_step_higher_threshold(
        self,
    ):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red") + group(
            "negative", "Keep order-red.", None
        )
        good = records_for(cases, positive=0.95)
        results = {
            "adapter-z": {"records": good, "step": 20},
            "base": {"records": good, "step": 0},
            "adapter-a": {"records": good, "step": 10},
        }
        selected = m.select_candidate(cases, results)
        self.assertEqual(selected["name"], "base")
        self.assertEqual(selected["step"], 0)
        self.assertEqual(selected["threshold"], 0.9)
        self.assertEqual(selected["metrics"]["raw_accuracy"], 1)
        self.assertEqual(selected["selection_split"], "dev")
        self.assertFalse(selected["independent_final_qualified"])
        self.assertEqual(len(selected["candidates"]), 15)
        self.assertEqual(
            selected, m.select_candidate(cases, dict(reversed(list(results.items()))))
        )
        for changed in [
            {"adapter": {"records": good, "step": 10}},
            {"base": {"records": good, "step": True}},
        ]:
            with self.assertRaises(ValueError):
                m.select_candidate(cases, changed)
        for thresholds in [[], [True], [float("nan")], [0.5, 0.5], [1.1]]:
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                m.select_candidate(cases, results, thresholds)
        with self.assertRaises(ValueError):
            m.select_candidate(cases, results, split="final")
        final_cases = group(
            "authorized", "Cancel order-red.", "order-red", split="final"
        )
        with self.assertRaises(ValueError):
            m.select_candidate(final_cases, results)

    def test_selection_metrics_outweigh_step_or_threshold(self):
        m = self.module()
        cases = group("authorized", "Cancel order-red.", "order-red") + group(
            "negative", "Keep order-red.", None
        )
        base = records_for(cases, positive=0.4, negative=0.1)
        adapted = records_for(cases, positive=0.75, negative=0.1)
        result = m.select_candidate(
            cases,
            {
                "base": {"records": base, "step": 0},
                "step-100": {"records": adapted, "step": 100},
            },
        )
        self.assertEqual((result["name"], result["threshold"]), ("step-100", 0.7))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
