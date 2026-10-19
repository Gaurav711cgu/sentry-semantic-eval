"""Admission provenance and terminal-failure handling without inference."""

import hashlib
import importlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class GuardTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("guards_v8")
        except ImportError:
            self.fail("separately versioned campaign provenance guards are missing")

    def fixture(self, stage="profile", *, failed=False):
        from campaign_v8 import digest_json, summarize

        path = ROOT / ("protocol_v8_" + stage + ".json")
        raw = path.read_bytes()
        p = json.loads(raw)
        run = 1 if stage == "profile" else 2
        grant = {
            "experiment_id": p["experiment_id"],
            "source_head": "1" * 40,
            "protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "calls": p["calls"],
            "campaign_id": "measured-v8-20261019",
            "stage": stage,
            "run_id": run,
            "attempt": 1,
            "status": "consumed_before_execution",
        }
        loaded = {
            "event": "loaded",
            "uid": 65534,
            "network_denied": True,
            "private_canary_denied": True,
            "dtype": "gguf_q4_k_m",
            "llama_cpp": "0.3.36",
            "numpy": "1.26.4",
            "threads": 4,
            "cpu_count": 4,
            "cpu_features": "avx2",
            "python": p["runtime"]["python"],
            "load_seconds": 1,
            "peak_rss_bytes": 1024,
        }
        events = [loaded]
        records = []
        for request in p["requests"]:
            result = {
                "event": "completed",
                "id": request["id"],
                "mode": request["mode"],
                "options": request["options"],
                "logits": [0.0] * len(request["options"]),
                "input_tokens": 10,
                "duration_seconds": 1,
                "peak_rss_bytes": 1024,
                "top_token_id": 32,
                "top_token_text": "A",
                "top_token_is_option": True,
            }
            events += [
                {"event": "started", "id": request["id"], "mode": request["mode"]},
                result,
            ]
            records.append(result)
        events.append({"event": "finished", "completed_calls": p["calls"]})
        report = {
            "execution_status": "completed",
            "evidence_status": "verified",
            "passed": False,
            "owned_worker_stopped": True,
            "temporary_artifacts_removed": True,
            "source_sha256": p["source_sha256"],
            "calls_consumed": p["calls"],
            "events": events,
            "stage": stage,
            "model": p["model"],
            "protocol_sha256": grant["protocol_sha256"],
            "grant": grant,
            **summarize(records, p, ROOT),
        }
        report["passed"] = report["stage_gate_passed"]
        if failed:
            report.update(
                execution_status="failed",
                evidence_status="failed",
                passed=False,
                failure_type="TimeoutError",
                failure_phase="inference",
                events=events[:2],
            )
        payload = json.dumps(report).encode()
        anchor = {
            "report_sha256": hashlib.sha256(payload).hexdigest(),
            "protocol_sha256": grant["protocol_sha256"],
            "grant_sha256": digest_json(grant),
            "run_id": run,
        }
        return path, p, grant, report, payload, anchor

    def verify(self, stage="profile", failed=False):
        m = self.module()
        path, _p, grant, _report, payload, anchor = self.fixture(stage, failed=failed)
        parsed = m.parse_bound_json(payload, anchor["report_sha256"])
        return m.verify_prerequisite(
            parsed, path, ROOT, stage=stage, anchor=anchor, ledger_grant=grant
        )

    def test_report_payload_is_pinned_before_json_parsing(self):
        m = self.module()
        _, _, _, report, payload, anchor = self.fixture()
        self.assertEqual(m.parse_bound_json(payload, anchor["report_sha256"]), report)
        for raw, digest in [
            (payload + b" ", anchor["report_sha256"]),
            (b"not-json", "0" * 64),
            (b"x" * (2 * 1024 * 1024 + 1), "1" * 64),
        ]:
            with self.assertRaises(ValueError):
                m.parse_bound_json(raw, digest)

    def test_complete_admission_recomputes_transcript_and_summary(self):
        result = self.verify()
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["performance_status"], "passed")

    def test_wrong_protocol_stage_model_runtime_grant_and_summary_are_rejected(self):
        m = self.module()
        path, _p, grant, report, _, anchor = self.fixture()
        variants = []
        for key, value in [
            ("protocol_sha256", "0" * 64),
            ("stage", "dev-b"),
            ("model", {"wrong": "model"}),
            ("grant", {**grant, "run_id": 999}),
            ("source_sha256", {}),
            ("calls_consumed", 99),
            ("passed", False),
        ]:
            variants.append({**report, key: value})
        changed = deepcopy(report)
        changed["events"][0]["python"] = "3.14.2"
        variants.append(changed)
        changed = deepcopy(report)
        changed["performance_by_candidate"]["a"]["max_forward_seconds"] = 0
        variants.append(changed)
        for changed in variants:
            with self.subTest(keys=changed.keys()), self.assertRaises(ValueError):
                m.verify_prerequisite(
                    changed,
                    path,
                    ROOT,
                    stage="profile",
                    anchor=anchor,
                    ledger_grant=grant,
                )
        for changed_anchor in [
            {**anchor, "protocol_sha256": "0" * 64},
            {**anchor, "grant_sha256": "0" * 64},
            {**anchor, "run_id": 999},
        ]:
            with self.assertRaises(ValueError):
                m.verify_prerequisite(
                    report,
                    path,
                    ROOT,
                    stage="profile",
                    anchor=changed_anchor,
                    ledger_grant=grant,
                )

    def test_only_authenticated_terminal_development_failure_can_be_excluded(self):
        result = self.verify("dev-a", failed=True)
        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(result["semantic_status"], "failed")
        self.assertFalse(result["passed"])
        m = self.module()
        path, _p, grant, report, _, anchor = self.fixture("dev-a", failed=True)
        for key, value in [
            ("execution_status", "running"),
            ("evidence_status", "unverified"),
            ("passed", True),
            ("failure_type", ""),
            ("owned_worker_stopped", False),
            ("temporary_artifacts_removed", False),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.verify_prerequisite(
                    {**report, key: value},
                    path,
                    ROOT,
                    stage="dev-a",
                    anchor=anchor,
                    ledger_grant=grant,
                )
        with self.assertRaises(ValueError):
            self.verify("profile", failed=True)

    def test_every_reserved_grant_has_exact_protocol_identity_and_consumed_state(self):
        m = self.module()
        _, _, grant, _, _, _ = self.fixture()
        ledger = {
            "campaign_id": "measured-v8-20261019",
            "budget_max": 296,
            "grants": {"profile": grant},
        }
        self.assertEqual(m.validate_ledger_strict(ledger, ROOT), 8)
        for key, value in [
            ("attempt", 2),
            ("status", "pending"),
            ("stage", "dev-b"),
            ("protocol_sha256", "0" * 64),
            ("source_head", "branch-name"),
            ("campaign_id", "another"),
            ("experiment_id", "another"),
        ]:
            changed = deepcopy(ledger)
            changed["grants"]["profile"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_ledger_strict(changed, ROOT)

    def test_guarded_runner_cannot_dispatch_another_profile(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_campaign_v8_guarded.py"),
                "--stage",
                "profile",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_failed_candidate_does_not_block_valid_passing_candidate_selection(self):
        from campaign_v8 import select_candidate

        failed = self.verify("dev-a", failed=True)
        passing = {
            "execution_status": "completed",
            "evidence_status": "verified",
            "semantic_status": "passed",
            "performance_status": "passed",
            "raw_summary": {"raw_accuracy": 1.0, "eligible_recall": 1.0},
            "max_forward_seconds": 1.0,
        }
        self.assertEqual(select_candidate({"a": failed, "b": passing}), "b")


if __name__ == "__main__":
    unittest.main()
