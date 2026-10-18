"""Isolated, offline, label-free CPU worker. Outputs scores, never effects."""

import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

from contracts import verify_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_model(args.model, p["model"]["files"])
    if os.getuid() != 65534 or any(
        key in os.environ for key in ["GITHUB_TOKEN", "GH_TOKEN", "DATABASE_URL"]
    ):
        raise ValueError("worker principal/credential boundary failed")
    try:
        args.canary.read_bytes()
    except PermissionError:
        denied = True
    else:
        raise ValueError("parent-private boundary failed")
    with socket.socket() as connection:
        connection.settimeout(0.2)
        try:
            connection.connect(("1.1.1.1", 443))
        except OSError:
            network = True
        else:
            raise ValueError("network boundary failed")
    import safetensors
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        sys.version.split()[0] != p["runtime"]["python"]
        or torch.__version__.split("+")[0] != p["runtime"]["torch"]
        or transformers.__version__ != p["runtime"]["transformers"]
        or safetensors.__version__ != p["runtime"]["safetensors"]
    ):
        raise ValueError("runtime drift")
    torch.set_num_threads(2)
    begun = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
    ).eval()
    loaded = time.monotonic() - begun
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(
        json.dumps(
            {
                "event": "loaded",
                "dtype": str(model.dtype).removeprefix("torch."),
                "uid": os.getuid(),
                "network_denied": network,
                "private_canary_denied": denied,
                "load_seconds": loaded,
                "peak_rss_bytes": peak,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": sys.version.split()[0],
            }
        ),
        flush=True,
    )
    if loaded > p["max_load_seconds"] or peak > p["max_rss_bytes"]:
        raise ValueError("load resource budget exceeded")
    for item in p["requests"]:
        print(
            json.dumps({"event": "started", "id": item["id"], "mode": item["mode"]}),
            flush=True,
        )
        inputs = tokenizer.apply_chat_template(
            item["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        count = inputs["input_ids"].shape[-1]
        if not 1 <= count <= 1024:
            raise ValueError("prompt token bound exceeded")
        token_ids = [
            tokenizer.encode(chr(65 + i), add_special_tokens=False)
            for i in range(len(item["options"]))
        ]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError("single-letter token identity changed")
        before = time.monotonic()
        with torch.inference_mode():
            full_logits = (
                model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            )
            logits = full_logits[[ids[0] for ids in token_ids]]
            top_id = int(full_logits.argmax().item())
        duration = time.monotonic() - before
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if (
            not torch.isfinite(logits).all()
            or duration > p["max_forward_seconds"]
            or peak > p["max_rss_bytes"]
        ):
            raise ValueError("forward output/resource gate failed")
        print(
            json.dumps(
                {
                    "event": "completed",
                    "id": item["id"],
                    "mode": item["mode"],
                    "options": item["options"],
                    "logits": logits.tolist(),
                    "top_token_id": top_id,
                    "top_token_text": tokenizer.decode([top_id]),
                    "top_token_is_option": top_id in [ids[0] for ids in token_ids],
                    "input_tokens": count,
                    "duration_seconds": duration,
                    "peak_rss_bytes": peak,
                },
                allow_nan=False,
            ),
            flush=True,
        )
    print(
        json.dumps({"event": "finished", "completed_calls": len(p["requests"])}),
        flush=True,
    )


if __name__ == "__main__":
    main()
