"""Fail-closed contracts for a public development comparison, never a final."""

import math
from collections import defaultdict


def validate_grant(grant, expected, run_id, attempt):
    fields = set(expected) | {"run_id", "attempt", "status"}
    if (
        type(grant) is not dict
        or set(grant) != fields
        or type(run_id) is not int
        or run_id <= 0
        or type(attempt) is not int
        or attempt != 1
        or type(grant["run_id"]) is not int
        or grant["run_id"] != run_id
        or type(grant["attempt"]) is not int
        or grant["attempt"] != 1
        or type(grant["calls"]) is not int
        or grant["status"] != "consumed_before_execution"
        or any(grant[key] != value for key, value in expected.items())
    ):
        raise ValueError("consumed grant identity mismatch")
    return grant


def validate_events(events, protocol):
    requests = protocol["requests"]
    if len(requests) != protocol["calls"] or len(events) != 2 * len(requests) + 2:
        raise ValueError("exact complete event count required")
    loaded, finished = events[0], events[-1]
    if (
        loaded.get("event") != "loaded"
        or loaded.get("uid") != 65534
        or loaded.get("network_denied") is not True
        or loaded.get("private_canary_denied") is not True
        or finished != {"event": "finished", "completed_calls": len(requests)}
    ):
        raise ValueError("worker lifecycle or isolation evidence failed")
    seen = set()
    records = []
    for index, request in enumerate(requests):
        key = request["id"], request["mode"]
        if key in seen:
            raise ValueError("duplicate reserved request")
        seen.add(key)
        start, result = events[1 + 2 * index : 3 + 2 * index]
        if start != {"event": "started", "id": key[0], "mode": key[1]}:
            raise ValueError("ordered call start required")
        if (
            set(result)
            != {
                "event",
                "id",
                "mode",
                "options",
                "logits",
                "input_tokens",
                "duration_seconds",
                "peak_rss_bytes",
            }
            or result["event"] != "completed"
            or (result["id"], result["mode"]) != key
            or result["options"] != request["options"]
            or type(result["logits"]) is not list
            or len(result["logits"]) != len(request["options"])
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in result["logits"]
            )
            or type(result["input_tokens"]) is not int
            or not 1 <= result["input_tokens"] <= 1024
            or type(result["duration_seconds"]) not in (int, float)
            or not math.isfinite(result["duration_seconds"])
            or not 0 <= result["duration_seconds"] <= protocol["max_forward_seconds"]
            or type(result["peak_rss_bytes"]) is not int
            or not 0 < result["peak_rss_bytes"] <= protocol["max_rss_bytes"]
        ):
            raise ValueError("call identity, output or resource contract failed")
        records.append(result)
    return records


def run_passed(*, complete, graded, stopped, removed, failure):
    return (
        all(value is True for value in [complete, graded, stopped, removed])
        and failure is None
    )


def grade(cases, predictions):
    ids = [case["id"] for case in cases]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("exact unique case prediction identities required")
    correct = eligible = hits = 0
    groups = defaultdict(list)
    for case in cases:
        target = predictions[case["id"]]
        correct += target == case["raw_target"]
        eligible += case["effect_target"] is not None
        hits += case["effect_target"] is not None and target == case["effect_target"]
        groups[case["family"]].append(target)
    return {
        "cases": len(cases),
        "raw_exact_correct": correct,
        "raw_accuracy": correct / len(cases),
        "eligible_targets": eligible,
        "eligible_raw_target_hits": hits,
        "eligible_recall": hits / eligible if eligible else None,
        "permutation_groups": len(groups),
        "consistent_groups": sum(len(set(v)) == 1 for v in groups.values()),
        "controller_validation_performed": False,
        "independent_final_qualified": False,
    }


def verify_model(root, files):
    import hashlib
    from pathlib import Path

    if type(files) is not dict or not 1 <= len(files) <= 16:
        raise ValueError("bounded exact model manifest required")
    if {p.name for p in root.iterdir()} != set(files):
        raise ValueError("model file inventory mismatch")
    for name, record in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe model filename")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or type(record["bytes"]) is not int
            or not 0 < record["bytes"] <= 4 * 1024**3
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("model file identity mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError("model digest mismatch")


def worker_error_tail(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2000))
        return stream.read(2000).decode("utf-8", errors="replace")[-2000:]
