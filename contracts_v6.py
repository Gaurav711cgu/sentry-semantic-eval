"""Separately versioned BF16 diagnostics; v5 admission/grading remain unchanged."""

import math
from copy import deepcopy

from contracts import validate_events as validate_base


def validate_events(events, protocol):
    if not events or events[0].get("dtype") != "bfloat16":
        raise ValueError("declared BF16 load evidence required")
    loaded = events[0]
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
