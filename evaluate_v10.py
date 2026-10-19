"""Independent, standard-library scoring of canonical candidate probabilities.

This module never runs inference or a controller, approves an action, or selects
on a final split. Raw semantic targets include absent and ineligible orders.
Only an expected target present with status exactly ``processing`` contributes
to the diagnostic eligible-target recall denominator. This is the canonical
PROCESSING_POLICY rule, also checked by position_qualification.py; ``pending``
is not an alias. No effect is authorized or observed by this calculation.
"""

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping

from pair_v10 import SYSTEM_PROMPT, choose_target, messages_for_pair, pair_inputs

ELIGIBLE_STATUSES = frozenset({"processing"})
ELIGIBILITY_RULE = (
    "expected_choice is non-null and identifies a supplied order with status "
    "exactly processing; diagnostic recall only, no controller validation"
)
DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
SCORER = "canonical-binary-candidate-v10"
DRAFT_FORWARD_LIMIT_SECONDS = 10.0
SELECTION_RULE = (
    "development only: descending raw_accuracy, eligible_recall, physical "
    "pair_balanced_accuracy; ascending checkpoint step; descending threshold; "
    "ascending candidate name as the final deterministic tie-break"
)


def _probability(value, description):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(description + " must be a finite probability")
    return float(value)


def _text(value, description, maximum=32768):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError("bounded nonempty " + description + " required")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("valid UTF-8 " + description + " required") from error
    return value


def _duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("duration_seconds must be finite and nonnegative")
    return float(value)


def validate_score_records(expectedcontents, records):
    """Require one finite score for every unique expected canonical prefix.

    Accepted worker records are a mapping from prefix to a record containing
    ``positive_probability`` and optional ``duration_seconds``, or a list of
    those records with an additional ``prefix_content`` field. A mapping of
    scalar probabilities is useful for untimed fixtures. Missing timing remains
    missing; it can never pass the draft latency gate. Duplicate list rows fail
    even if values match. Duplicate keys already collapsed by a JSON decoder
    cannot be recovered here: decode external JSON with duplicate-key rejection.
    """
    if isinstance(expectedcontents, (str, bytes, Mapping)):
        raise ValueError("collection of unique canonical prefixes required")  # noqa: TRY004 - uniform invalid-evidence contract
    try:
        expected = list(expectedcontents)
    except TypeError as error:
        raise ValueError("collection of unique canonical prefixes required") from error
    for content in expected:
        _text(content, "canonical prefix")
        messages_for_pair(content)
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected canonical prefix")
    if isinstance(records, Mapping):
        pairs = list(records.items())
    elif type(records) is list:
        pairs = []
        for row in records:
            if type(row) is not dict or "prefix_content" not in row:
                raise ValueError("worker rows require prefix_content")
            pairs.append(
                (
                    row["prefix_content"],
                    {
                        key: value
                        for key, value in row.items()
                        if key != "prefix_content"
                    },
                )
            )
    else:
        raise ValueError("score mapping or unique worker row list required")
    normalized = {}
    for content, record in pairs:
        _text(content, "record prefix")
        if content in normalized:
            raise ValueError("duplicate score record prefix")
        if type(record) in (int, float):
            record = {"positive_probability": record}
        if type(record) is not dict or set(record) not in (
            {"positive_probability"},
            {"positive_probability", "duration_seconds"},
        ):
            raise ValueError("exact probability and optional duration fields required")
        row = {
            "positive_probability": _probability(
                record["positive_probability"], "positive_probability"
            )
        }
        if "duration_seconds" in record:
            row["duration_seconds"] = _duration(record["duration_seconds"])
        normalized[content] = row
    if set(normalized) != set(expected):
        raise ValueError(
            "score records must exactly cover expected prefixes without extras"
        )
    return normalized


def cache_identity(model, runtime, adapter, prefix_content):
    """Hash the exact computation identity, including the frozen system prompt.

    Callers supply model revision/file hashes, runtime identity and adapter hash
    (or ``None`` for the base control). This hash binds those declared identities;
    it does not verify any model files, adapter weights or worker provenance.
    Thresholds and gold labels are deliberately absent from score identities.
    """
    _text(prefix_content, "canonical prefix")
    messages_for_pair(prefix_content)
    if (
        not isinstance(model, Mapping)
        or not model
        or not isinstance(runtime, Mapping)
        or not runtime
    ):
        raise ValueError("nonempty model and runtime identities required")
    if adapter is not None and (not isinstance(adapter, Mapping) or not adapter):
        raise ValueError("nonempty adapter identity or null base control required")
    value = {
        "scorer": SCORER,
        "model": dict(model),
        "runtime": dict(runtime),
        "adapter": dict(adapter) if adapter is not None else None,
        "system_prompt": SYSTEM_PROMPT,
        "prefix_content": prefix_content,
        "positive_label": "A",
        "negative_label": "B",
    }
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("finite JSON computation identities required") from error
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prepare_cases(cases):
    if type(cases) is not list or not cases:
        raise ValueError("nonempty case list required")
    ids, groups, prepared, physical_labels = set(), defaultdict(list), [], {}
    for case in cases:
        if type(case) is not dict or set(case) != {
            "id",
            "group_id",
            "kind",
            "request",
            "orders",
            "expected_choice",
        }:
            raise ValueError("exact v10 case fields required")
        for key in ("id", "group_id", "kind"):
            _text(case[key], key, 512)
        if case["id"] in ids:
            raise ValueError("unique case IDs required")
        ids.add(case["id"])
        orders = case["orders"]
        if type(orders) is not list or len(orders) != 3:
            raise ValueError("three snapshots required for six-view permutation groups")
        for order in orders:
            if type(order) is not dict or set(order) != {"id", "status"}:
                raise ValueError("exact order id/status fields required")
            _text(order["status"], "order status", 128)
        rows = pair_inputs(case["request"], [order["id"] for order in orders])
        target = case["expected_choice"]
        if target is not None and (
            type(target) is not str or target not in {row["candidate"] for row in rows}
        ):
            raise ValueError(
                "expected raw target must belong to listed/referenced union or be null"
            )
        eligible = target is not None and any(
            order["id"] == target and order["status"] in ELIGIBLE_STATUSES
            for order in orders
        )
        for row in rows:
            positive = row["candidate"] == target
            content = row["prefix_content"]
            if content in physical_labels and physical_labels[content] != positive:
                raise ValueError("canonical prefix has contradictory gold pair labels")
            physical_labels[content] = positive
        prepared.append((case, rows, eligible))
        groups[case["group_id"]].append(case)
    for group in groups.values():
        reference = group[0]
        snapshots = sorted(
            (order["id"], order["status"]) for order in reference["orders"]
        )
        if len(group) != 6:
            raise ValueError("each group must have exactly six permutation views")
        for case in group:
            if any(
                case[key] != reference[key]
                for key in ("request", "expected_choice", "kind")
            ):
                raise ValueError(
                    "permutations must preserve request, kind and gold target"
                )
            if (
                sorted((order["id"], order["status"]) for order in case["orders"])
                != snapshots
            ):
                raise ValueError("permutations must preserve snapshot IDs and statuses")
        actual = {tuple(order["id"] for order in case["orders"]) for case in group}
        expected = set(
            itertools.permutations(order["id"] for order in reference["orders"])
        )
        if len(actual) != 6 or actual != expected:
            raise ValueError("all six unique snapshot permutations required")
    return prepared, groups, physical_labels


def _confusion():
    return {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }


def _add_pair(counts, expected, predicted):
    key = ("true_" if expected == predicted else "false_") + (
        "positive" if predicted else "negative"
    )
    counts[key] += 1


def _pair_metrics(counts):
    tp, fp, tn, fn = (
        counts[key]
        for key in (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        )
    )
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "positive_recall": sensitivity,
        "negative_recall": specificity,
    }


def score_cases(cases, records, threshold):
    """Grade frozen raw choices and pair scores without running action policy.

    Physical pair metrics count each canonical prefix once. Logical pair metrics
    include every case/candidate binding and disclose cache aliases explicitly.
    A per-draft latency is the sum of all candidate forwards, even when several
    bindings reuse a cached canonical computation. It is a conservative uncached
    estimate from measured pair times, not an observed end-to-end product trace.
    """
    threshold = _probability(threshold, "threshold")
    prepared, groups, physical_labels = _prepare_cases(cases)
    scores = validate_score_records(sorted(physical_labels), records)
    physical_counts, logical_counts = _confusion(), _confusion()
    for content, expected in physical_labels.items():
        _add_pair(
            physical_counts,
            expected,
            scores[content]["positive_probability"] >= threshold,
        )
    choices, outcomes, draft_times = {}, [], []
    correct = eligible_count = eligible_hits = logical_calls = 0
    false_escalations = wrong_selected_targets = 0
    all_times = all("duration_seconds" in record for record in scores.values())
    by_kind = defaultdict(
        lambda: {
            "cases": 0,
            "raw_exact_correct": 0,
            "eligible_targets": 0,
            "eligible_raw_target_hits": 0,
        }
    )
    for case, rows, eligible in prepared:
        selected = choose_target(
            [
                {
                    "candidate": row["candidate"],
                    "positive_probability": scores[row["prefix_content"]][
                        "positive_probability"
                    ],
                }
                for row in rows
            ],
            threshold,
        )
        choices[case["id"]] = selected
        exact = selected == case["expected_choice"]
        hit = eligible and exact
        correct += exact
        eligible_count += eligible
        eligible_hits += hit
        logical_calls += len(rows)
        false_escalations += case["expected_choice"] is not None and selected is None
        wrong_selected_targets += selected is not None and not exact
        timing = (
            sum(scores[row["prefix_content"]]["duration_seconds"] for row in rows)
            if all_times
            else None
        )
        if timing is not None:
            draft_times.append(timing)
        for row in rows:
            _add_pair(
                logical_counts,
                row["candidate"] == case["expected_choice"],
                scores[row["prefix_content"]]["positive_probability"] >= threshold,
            )
        outcomes.append(
            {
                "id": case["id"],
                "group_id": case["group_id"],
                "kind": case["kind"],
                "expected_choice": case["expected_choice"],
                "selected_choice": selected,
                "raw_exact_correct": exact,
                "eligible_target": eligible,
                "eligible_raw_target_hit": hit,
                "candidate_count": len(rows),
                "draft_forward_seconds": timing,
            }
        )
        totals = by_kind[case["kind"]]
        totals["cases"] += 1
        totals["raw_exact_correct"] += exact
        totals["eligible_targets"] += eligible
        totals["eligible_raw_target_hits"] += hit
    physical_metrics, logical_metrics = (
        _pair_metrics(physical_counts),
        _pair_metrics(logical_counts),
    )
    group_results = [
        {
            "group_id": group_id,
            "views": len(group),
            "consistent": len({choices[case["id"]] for case in group}) == 1,
            "exact": all(
                choices[case["id"]] == case["expected_choice"] for case in group
            ),
        }
        for group_id, group in sorted(groups.items())
    ]
    timings = (
        [record["duration_seconds"] for record in scores.values()] if all_times else []
    )
    maximum = max(draft_times) if draft_times else None
    return {
        "schema": "canonical-pair-score-v10",
        "threshold": threshold,
        "cases": len(cases),
        "raw_exact_correct": correct,
        "raw_accuracy": correct / len(cases),
        "eligible_targets": eligible_count,
        "eligible_raw_target_hits": eligible_hits,
        "eligible_recall": eligible_hits / eligible_count if eligible_count else None,
        "eligibility_rule": ELIGIBILITY_RULE,
        "eligible_statuses": sorted(ELIGIBLE_STATUSES),
        "false_escalations": false_escalations,
        "wrong_selected_targets": wrong_selected_targets,
        "permutation_groups": len(groups),
        "consistent_groups": sum(row["consistent"] for row in group_results),
        "exact_groups": sum(row["exact"] for row in group_results),
        "invariance_scope": "structural canonicalization, not learned permutation robustness",
        "physical_scoring_calls": len(scores),
        "logical_scoring_calls": logical_calls,
        "cache_aliases": logical_calls - len(scores),
        "pair_counts": physical_counts,
        "logical_pair_counts": logical_counts,
        "pair_balanced_accuracy": physical_metrics["balanced_accuracy"],
        "pair_precision": physical_metrics["precision"],
        "pair_positive_recall": physical_metrics["positive_recall"],
        "pair_negative_recall": physical_metrics["negative_recall"],
        "logical_pair_balanced_accuracy": logical_metrics["balanced_accuracy"],
        "logical_pair_precision": logical_metrics["precision"],
        "timing_complete": all_times,
        "physical_forward_seconds": sum(timings) if all_times else None,
        "logical_forward_seconds": sum(draft_times) if all_times else None,
        "max_pair_forward_seconds": max(timings) if timings else None,
        "max_draft_forward_seconds": maximum,
        "draft_forward_limit_seconds": DRAFT_FORWARD_LIMIT_SECONDS,
        "draft_latency_gate_passed": maximum <= DRAFT_FORWARD_LIMIT_SECONDS
        if maximum is not None
        else None,
        "latency_scope": "sum of all candidate pair forwards per draft, including cache aliases; conservative uncached estimate",
        "choices": choices,
        "case_results": outcomes,
        "group_results": group_results,
        "by_kind": dict(sorted(by_kind.items())),
        "controller_validation_performed": False,
        "independent_final_qualified": False,
    }


def select_candidate(cases, results, thresholds=DEFAULT_THRESHOLDS, *, split="dev"):
    """Choose a checkpoint/threshold on development data only, including step 0.

    Call final evaluation with ``score_cases`` and the previously selected fixed
    threshold. The selector rejects final/train identities, requires development
    case/group prefixes and includes a base step-zero control in its ranking.
    It does not confer a semantic or latency qualification on the selected item.
    """
    if split != "dev":
        raise ValueError("checkpoint and threshold selection is development-only")
    _prepare_cases(cases)
    if any(
        not re.match(r"^dev(?:[-/:_]|$)", case[key])
        for case in cases
        for key in ("id", "group_id")
    ):
        raise ValueError("explicit development case and group identities required")
    if type(results) is not dict or not results:
        raise ValueError(
            "nonempty candidate result mapping including step-zero base required"
        )
    if type(thresholds) not in (tuple, list) or not thresholds:
        raise ValueError("nonempty unique threshold list required")
    checked_thresholds = [_probability(value, "threshold") for value in thresholds]
    if len(set(checked_thresholds)) != len(checked_thresholds):
        raise ValueError("duplicate selection threshold")
    has_base, candidates = False, []
    for name, result in sorted(results.items()):
        _text(name, "candidate name", 512)
        if type(result) is not dict or set(result) != {"records", "step"}:
            raise ValueError("exact candidate records/step fields required")
        step = result["step"]
        if type(step) is not int or step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        has_base = has_base or step == 0
        for threshold in sorted(checked_thresholds):
            candidates.append(
                {
                    "name": name,
                    "step": step,
                    "threshold": threshold,
                    "metrics": score_cases(cases, result["records"], threshold),
                }
            )
    if not has_base:
        raise ValueError("step-zero base control must be included")

    def rank(candidate):
        metrics = candidate["metrics"]
        return (
            -metrics["raw_accuracy"],
            -(
                metrics["eligible_recall"]
                if metrics["eligible_recall"] is not None
                else -1
            ),
            -(
                metrics["pair_balanced_accuracy"]
                if metrics["pair_balanced_accuracy"] is not None
                else -1
            ),
            candidate["step"],
            -candidate["threshold"],
            candidate["name"],
        )

    candidates.sort(key=rank)
    selected = candidates[0]
    return {
        **selected,
        "selection_split": "dev",
        "selection_rule": SELECTION_RULE,
        "candidates": candidates,
        "controller_validation_performed": False,
        "independent_final_qualified": False,
    }
