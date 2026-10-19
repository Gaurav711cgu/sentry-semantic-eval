"""Versioned prerequisite provenance guards; consumed v8 experiments stay frozen."""

import hashlib
import json
import re
from pathlib import Path

from campaign_v8 import STAGE_CALLS, digest_json, validate_ledger, verify_result
from contracts import validate_grant

MAX_REPORT_BYTES = 2 * 1024 * 1024
CAMPAIGN_ID = "measured-v8-20261019"


def _sha(value, length=64):
    return (
        type(value) is str
        and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None
    )


def parse_bound_json(payload, expected_sha256):
    """Bind exact fetched bytes before any JSON interpretation."""
    if (
        type(payload) is not bytes
        or len(payload) > MAX_REPORT_BYTES
        or not _sha(expected_sha256)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("prerequisite report payload integrity mismatch")
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid bound report JSON") from error
    if type(value) is not dict:
        raise ValueError("bound report must be an object")
    return value


def _protocol(path, *, stage, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("prerequisite protocol must be a regular file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("prerequisite protocol integrity mismatch")
    p = json.loads(raw)
    if p.get("stage") != stage or p.get("calls") != STAGE_CALLS[stage]:
        raise ValueError("prerequisite protocol stage/call identity mismatch")
    return p, digest


def _validate_stage_grant(grant, protocol, protocol_sha256, *, stage):
    if type(grant) is not dict or not _sha(grant.get("source_head"), 40):
        raise ValueError("exact grant source commit required")
    expected = {
        "experiment_id": protocol["experiment_id"],
        "source_head": grant["source_head"],
        "protocol_sha256": protocol_sha256,
        "calls": STAGE_CALLS[stage],
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
    }
    validate_grant(grant, expected, grant.get("run_id"), 1)


def validate_ledger_strict(ledger, root):
    total = validate_ledger(ledger)
    for stage, grant in ledger["grants"].items():
        p, digest = _protocol(
            Path(root) / ("protocol_v8_" + stage + ".json"), stage=stage
        )
        _validate_stage_grant(grant, p, digest, stage=stage)
    return total


def verify_prerequisite(report, protocol_path, root, *, stage, anchor, ledger_grant):
    """Verify a hash-bound report; terminal development failures are ineligible.

    The caller must use parse_bound_json on fetched bytes before calling this
    function. Complete results are regraded by the original frozen contracts.
    """
    if (
        type(anchor) is not dict
        or set(anchor) != {"report_sha256", "protocol_sha256", "grant_sha256", "run_id"}
        or any(
            not _sha(anchor[k])
            for k in ["report_sha256", "protocol_sha256", "grant_sha256"]
        )
        or type(anchor["run_id"]) is not int
        or anchor["run_id"] <= 0
    ):
        raise ValueError("exact prerequisite report/protocol/grant anchors required")
    p, digest = _protocol(
        protocol_path, stage=stage, expected_sha256=anchor["protocol_sha256"]
    )
    _validate_stage_grant(ledger_grant, p, digest, stage=stage)
    if (
        digest_json(ledger_grant) != anchor["grant_sha256"]
        or ledger_grant["run_id"] != anchor["run_id"]
        or type(report) is not dict
        or report.get("grant") != ledger_grant
        or report.get("stage") != stage
        or report.get("model") != p["model"]
        or report.get("protocol_sha256") != digest
        or report.get("source_sha256") != p["source_sha256"]
        or report.get("calls_consumed") != p["calls"]
    ):
        raise ValueError("prerequisite stage/model/protocol/grant provenance mismatch")
    if report.get("execution_status") == "completed":
        events = report.get("events")
        if (
            type(events) is not list
            or not events
            or type(events[0]) is not dict
            or any(events[0].get(key) != value for key, value in p["runtime"].items())
        ):
            raise ValueError("observed prerequisite runtime identity mismatch")
        expected = verify_result(report, p, root)
        if report.get("passed") is not expected["stage_gate_passed"]:
            raise ValueError("top-level pass contradicts recomputed stage gate")
        return report
    if (
        stage.startswith("dev-")
        and report.get("execution_status") == "failed"
        and report.get("evidence_status") == "failed"
        and report.get("passed") is False
        and type(report.get("failure_type")) is str
        and 1 <= len(report["failure_type"]) <= 128
        and report["failure_type"].strip()
        and report.get("owned_worker_stopped") is True
        and report.get("temporary_artifacts_removed") is True
    ):
        # Partial outputs do not enter ranking, even if a producer left metrics.
        return {**report, "semantic_status": "failed", "performance_status": "failed"}
    raise ValueError(
        "complete admission or authenticated terminal development failure required"
    )
