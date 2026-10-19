"""One prospectively fixed synthetic repair candidate; never a v8 promotion."""

import copy
import hashlib
import json
from pathlib import Path

from campaign_v8 import deduplicate
from campaign_v8 import summarize as summarize_v8
from contracts_v7 import validate_events
from guards_v8 import validate_ledger_strict

PROMPT = (
    "Extract CURRENT cancellation intent, not execution eligibility. "
    "Pick the named target only for a direct, unconditional request to cancel exactly one order NOW. "
    "An absent ID remains a target; status and policy come later. "
    "ESCALATE for negation, quoted/reported instructions, conditions, questions, revoked consent, "
    "multiple/unclear targets or policy/authority overrides. Keeping a different order is allowed. "
    "Examples: 'cancel id9' (absent) => id9; 'cancel id9 if X' => ESCALATE; "
    "'he wrote cancel id9' => ESCALATE; 'fee to cancel id9?' => ESCALATE; "
    "'keep id8; cancel id9' => id9. Reply with ONE option letter only."
)
STAGE_CALLS = {"repair-c": 156}
CAMPAIGN_ID = "repair-v9-20261019"


def build_protocol(root):
    p = copy.deepcopy(
        json.loads((Path(root) / "protocol_v8_confirm-a.json").read_text())
    )
    p.update(
        experiment_id="single-candidate-synthetic-repair-v9-20261019",
        stage="repair-c",
        candidate="c",
        v8_promotion_eligible=False,
        independent_final_qualified=False,
        independent_validation_status="not_claimed",
        max_worker_wall_seconds=1900,
        max_forward_seconds=10,
    )
    p.pop("prerequisite_reports", None)
    p["source_sha256"] = {}
    for r in p["logical_requests"]:
        r["messages"][0]["content"] = PROMPT
    p["requests"], p["bindings"] = deduplicate(
        p["logical_requests"], p["model"], p["runtime"]
    )
    p["method"] = (
        "One new candidate fixed before scoring the previously unscored AI-authored "
        "144-view synthetic benchmark, with12 distinct offline calibration computations. "
        "No v8 selection eligibility or independent validation is claimed; "
        "no reference, permutation, scoring method or quality threshold changes."
    )
    p["first_four_completed_requests_are_resource_admission"] = True
    p["prior_calls_reserved"] = 140
    p["total_budget_ceiling"] = 296
    return p


def validate_reservation(ledger, grant, protocol, root):
    prior_path = Path(root) / "campaign-v8-ledger.json"
    prior = json.loads(prior_path.read_text())
    previous = validate_ledger_strict(prior, root)
    if (
        set(ledger)
        != {
            "campaign_id",
            "budget_max",
            "prior_v8_ledger_sha256",
            "prior_calls_reserved",
            "grant",
        }
        or ledger["campaign_id"] != CAMPAIGN_ID
        or ledger["budget_max"] != 296
        or ledger["prior_calls_reserved"] != previous
        or previous != 140
        or ledger["prior_v8_ledger_sha256"]
        != hashlib.sha256(prior_path.read_bytes()).hexdigest()
        or ledger["grant"] != grant
        or grant.get("calls") != 156
        or protocol["calls"] != 156
        or len(protocol["requests"]) != 156
    ):
        raise ValueError("exact one-use reservation and unchanged prior140 required")
    return previous + grant["calls"]


def summarize(records, protocol, root):
    result = summarize_v8(records, protocol, root)
    result.update(
        v8_promotion_eligible=False,
        scope="Prospectively fixed single-candidate synthetic repair benchmark; actual controller audit required; independent validation not claimed",
    )
    return result


def verify_result(report, protocol, root):
    if (
        report.get("execution_status") != "completed"
        or report.get("evidence_status") != "verified"
        or report.get("owned_worker_stopped") is not True
        or report.get("temporary_artifacts_removed") is not True
        or report.get("source_sha256") != protocol["source_sha256"]
        or report.get("calls_consumed") != 156
        or report.get("completed_calls") != 156
        or report.get("model") != protocol["model"]
        or report.get("v8_promotion_eligible") is not False
        or report.get("independent_final_qualified") is not False
    ):
        raise ValueError("complete exact-source repair evidence required")
    records = validate_events(report["events"], protocol)
    expected = summarize(records, protocol, root)
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("repair summary differs from retained computations")
    return expected
