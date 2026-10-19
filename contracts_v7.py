"""Separately versioned GGUF diagnostics; v5 admission/grading remain unchanged."""

import math
from copy import deepcopy

from contracts import validate_events as validate_base


def profile(records, protocol):
    if protocol["calls"] != 4 or len(records) != 4:
        raise ValueError("complete four-request profile required")
    maximum = max(row["duration_seconds"] for row in records)
    return {
        "performance_gate_passed": maximum <= protocol["profile_max_forward_seconds"],
        "semantic_quality_evaluated": False,
        "max_forward_seconds": maximum,
        "max_rss_bytes": max(row["peak_rss_bytes"] for row in records),
    }


def validate_events(events, protocol):
    if not events or events[0].get("dtype") != "gguf_q4_k_m":
        raise ValueError("declared GGUF load evidence required")
    loaded = events[0]
    if (
        loaded.get("llama_cpp") != "0.3.36"
        or loaded.get("numpy") != "1.26.4"
        or loaded.get("threads") != 4
        or type(loaded.get("cpu_count")) is not int
        or not 1 <= loaded["cpu_count"] <= 256
        or type(loaded.get("cpu_features")) is not str
        or not 1 <= len(loaded["cpu_features"]) <= 4096
    ):
        raise ValueError("observed CPU runtime evidence required")
    seconds, peak = loaded.get("load_seconds"), loaded.get("peak_rss_bytes")
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 <= seconds <= protocol["max_load_seconds"]
        or type(peak) is not int
        or not 0 < peak <= protocol["max_rss_bytes"]
    ):
        raise ValueError("observed load resource gate failed")
    sanitized = deepcopy(events)
    for row in sanitized:
        if row.get("event") == "completed":
            token = row.get("top_token_id")
            text = row.get("top_token_text")
            is_option = row.get("top_token_is_option")
            if (
                type(token) is not int
                or not 0 <= token <= 200000
                or type(text) is not str
                or len(text) > 128
                or type(is_option) is not bool
            ):
                raise ValueError("full-vocabulary diagnostics required")
            for key in ["top_token_id", "top_token_text", "top_token_is_option"]:
                del row[key]
    validate_base(sanitized, protocol)
    return [row for row in events if row.get("event") == "completed"]
