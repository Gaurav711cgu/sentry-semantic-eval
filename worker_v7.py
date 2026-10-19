"""Label-free CPU GGUF scoring. No generation, customer effects or network."""

import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

from contracts import verify_model
from prompts_v7 import render


def emit(event):
    print(json.dumps(event, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_model(args.model, p["model"]["files"])
    if os.getuid() != 65534 or any(
        k in os.environ for k in ["GITHUB_TOKEN", "GH_TOKEN", "DATABASE_URL"]
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
    import llama_cpp
    import numpy as np

    if {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "llama_cpp": llama_cpp.__version__,
    } != p["runtime"]:
        raise ValueError("runtime drift")
    begun = time.monotonic()
    llm = llama_cpp.Llama(
        model_path=str(args.model / p["model"]["weights_file"]),
        n_ctx=1024,
        n_batch=1024,
        n_ubatch=512,
        n_threads=4,
        n_threads_batch=4,
        n_gpu_layers=0,
        offload_kqv=False,
        flash_attn=False,
        use_mmap=True,
        use_mlock=False,
        logits_all=False,
        seed=20261019,
        verbose=False,
    )
    if (
        llm.metadata.get("general.architecture") != "qwen3"
        or str(llm.metadata.get("general.file_type")) != "15"
    ):
        raise ValueError("GGUF architecture/quantization mismatch")
    loaded = time.monotonic() - begun
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    flags = next(
        (
            line.partition(":")[2].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("flags")
        ),
        "",
    )
    emit(
        {
            "event": "loaded",
            "dtype": "gguf_q4_k_m",
            "uid": os.getuid(),
            "network_denied": network,
            "private_canary_denied": denied,
            "load_seconds": loaded,
            "peak_rss_bytes": peak,
            "llama_cpp": llama_cpp.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "threads": 4,
            "cpu_count": len(os.sched_getaffinity(0)),
            "cpu_features": flags[:4096],
        }
    )
    if loaded > p["max_load_seconds"] or peak > p["max_rss_bytes"]:
        raise ValueError("load resource budget exceeded")
    for item in p["requests"]:
        emit({"event": "started", "id": item["id"], "mode": item["mode"]})
        before = time.monotonic()
        tokens = llm.tokenize(
            render(item["messages"]).encode(), add_bos=False, special=True
        )
        option_ids = [
            llm.tokenize(chr(65 + i).encode(), add_bos=False, special=False)
            for i in range(len(item["options"]))
        ]
        if not 1 <= len(tokens) <= 1024 or any(len(ids) != 1 for ids in option_ids):
            raise ValueError("prompt/option token identity bound exceeded")
        llm.reset()
        llm.eval(tokens)
        # Pinned0.3.36 does not fill Python scores when logits_all=False.
        # Its requested final-token logits remain in the native context buffer.
        logits = np.ctypeslib.as_array(
            llm._ctx.get_logits(), shape=(llm.n_vocab(),)
        ).copy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite model logits")
        indices = [ids[0] for ids in option_ids]
        top = int(logits.argmax())
        duration = time.monotonic() - before
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if duration > p["max_forward_seconds"] or peak > p["max_rss_bytes"]:
            raise ValueError("forward resource gate failed")
        emit(
            {
                "event": "completed",
                "id": item["id"],
                "mode": item["mode"],
                "options": item["options"],
                "logits": logits[indices].tolist(),
                "top_token_id": top,
                "top_token_text": llm.detokenize([top]).decode(errors="replace"),
                "top_token_is_option": top in indices,
                "input_tokens": len(tokens),
                "duration_seconds": duration,
                "peak_rss_bytes": peak,
            }
        )
    llm.close()
    emit({"event": "finished", "completed_calls": len(p["requests"])})


if __name__ == "__main__":
    main()
