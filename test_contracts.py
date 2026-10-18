"""Admission and transcript contracts must pass before any model is loaded."""

import importlib
import unittest
from copy import deepcopy


class Contracts(unittest.TestCase):
    def contract(self):
        try:
            return importlib.import_module("contracts")
        except ImportError:
            self.fail("Evaluation admission/transcript contracts are missing")

    def protocol(self):
        return {
            "experiment_id": "fixture-v5",
            "calls": 2,
            "source_sha256": {"worker.py": "a" * 64},
            "max_forward_seconds": 60,
            "max_rss_bytes": 8 * 1024**3,
            "requests": [
                {"id": "a", "mode": "raw", "options": [None, "order-red"]},
                {"id": "a", "mode": "content_free", "options": [None, "order-red"]},
            ],
        }

    def events(self):
        events = [
            {
                "event": "loaded",
                "uid": 65534,
                "network_denied": True,
                "private_canary_denied": True,
            }
        ]
        for mode in ["raw", "content_free"]:
            events.extend(
                [
                    {"event": "started", "id": "a", "mode": mode},
                    {
                        "event": "completed",
                        "id": "a",
                        "mode": mode,
                        "options": [None, "order-red"],
                        "logits": [1.0, 2.0],
                        "input_tokens": 20,
                        "duration_seconds": 0.2,
                        "peak_rss_bytes": 1000,
                    },
                ]
            )
        return events + [{"event": "finished", "completed_calls": 2}]

    def test_missing_or_duplicate_calls_are_rejected(self):
        c = self.contract()
        self.assertEqual(len(c.validate_events(self.events(), self.protocol())), 2)
        for events in [
            self.events()[:-1],
            self.events() + [self.events()[-1]],
            self.events()[:3] + self.events()[1:3] + self.events()[3:],
        ]:
            with self.assertRaises(ValueError):
                c.validate_events(events, self.protocol())

    def test_options_order_resources_and_nonfinite_scores_fail(self):
        c = self.contract()
        for change in [
            lambda x: x[2].update(options=["order-red", None]),
            lambda x: x[2].update(logits=[float("nan"), 1]),
            lambda x: x[2].update(duration_seconds=61),
            lambda x: x[2].update(peak_rss_bytes=0),
            lambda x: x[0].update(uid=0),
        ]:
            events = deepcopy(self.events())
            change(events)
            with self.assertRaises(ValueError):
                c.validate_events(events, self.protocol())

    def test_grant_binds_full_attempt_before_inference(self):
        c = self.contract()
        expected = {
            "experiment_id": "v5",
            "source_head": "a" * 40,
            "protocol_sha256": "b" * 64,
            "calls": 108,
        }
        grant = {
            **expected,
            "run_id": 123,
            "attempt": 1,
            "status": "consumed_before_execution",
        }
        self.assertEqual(c.validate_grant(grant, expected, 123, 1), grant)
        for changes in [
            {"attempt": 2},
            {"run_id": 124},
            {"calls": True},
            {"status": "pending"},
            {"extra": 1},
        ]:
            with self.assertRaises(ValueError):
                c.validate_grant({**grant, **changes}, expected, 123, 1)

    def test_grading_and_cleanup_are_separate_from_quality(self):
        c = self.contract()
        self.assertTrue(
            c.run_passed(
                complete=True, graded=True, stopped=True, removed=True, failure=None
            )
        )
        for key in ["complete", "graded", "stopped", "removed"]:
            values = dict(
                complete=True, graded=True, stopped=True, removed=True, failure=None
            )
            values[key] = False
            self.assertFalse(c.run_passed(**values))
        self.assertFalse(
            c.run_passed(
                complete=True, graded=True, stopped=True, removed=True, failure="error"
            )
        )

    def test_grade_requires_exact_case_identity_and_keeps_baselines(self):
        c = self.contract()
        cases = [
            {
                "id": "one",
                "family": "f",
                "raw_target": "order-red",
                "effect_target": "order-red",
                "processing": ["order-red"],
                "first_listed": "order-red",
            }
        ]
        result = c.grade(cases, {"one": "order-red"})
        self.assertEqual(result["raw_exact_correct"], 1)
        self.assertEqual(result["eligible_recall"], 1.0)
        with self.assertRaises(ValueError):
            c.grade(cases, {})
        with self.assertRaises(ValueError):
            c.grade(cases, {"one": "order-red", "extra": None})


class ModelFiles(unittest.TestCase):
    def test_model_files_require_exact_digest_and_no_links(self):
        import tempfile
        import hashlib
        from pathlib import Path

        c = importlib.import_module("contracts")
        self.assertTrue(
            hasattr(c, "verify_model"), "Exact model-file verification is missing"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_bytes(b"{}")
            manifest = {
                "config.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
            }
            c.verify_model(root, manifest)
            (root / "config.json").write_bytes(b"xx")
            with self.assertRaises(ValueError):
                c.verify_model(root, manifest)
            (root / "config.json").unlink()
            (root / "config.json").symlink_to(root / "outside")
            with self.assertRaises(ValueError):
                c.verify_model(root, manifest)


class ParentAdmission(unittest.TestCase):
    def test_reused_attempt_rejects_before_acquisition(self):
        import subprocess
        import sys
        import tempfile
        import json
        import os
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_probe.py")),
                    "--output",
                    str(output),
                ],
                env={**os.environ, "GITHUB_RUN_ATTEMPT": "2"},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(
                output.exists(),
                "Parent must persist a failed admission report before any acquisition",
            )
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["model_acquisition_started"])
            self.assertEqual(report["calls_consumed"], 0)


class Diagnostics(unittest.TestCase):
    def test_worker_error_tail_is_bounded(self):
        import tempfile
        from pathlib import Path

        c = importlib.import_module("contracts")
        self.assertTrue(
            hasattr(c, "worker_error_tail"), "Bounded worker diagnostics missing"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stderr"
            path.write_bytes(b"x" * 10000 + b"public failure")
            result = c.worker_error_tail(path)
            self.assertLessEqual(len(result), 2000)
            self.assertTrue(result.endswith("public failure"))


if __name__ == "__main__":
    unittest.main()
