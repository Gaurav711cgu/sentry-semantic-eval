"""Bounded CPU FP32 adapter training/scoring; no final labels enter this worker.

Class0=A=authorized; class1=B=not authorized. Inputs are prefix-only canonical
pair content. Checkpoint/threshold selection and final grading belong to the
separate parent. Model/tokenizer files must already exist at the pinned path.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

from pair_v10 import messages_for_pair

SEED = 1019
THREADS = 4
MAX_PREFIX_TOKENS = 384
STEPS = 256
MICROBATCH = 2
ACCUMULATION = 4
MAX_RSS = 8 * 1024**3
MAX_WALL = 5400


def label_ids(tokenizer):
    labels = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")
    ]
    if (
        any(
            type(ids) is not list or len(ids) != 1 or type(ids[0]) is not int
            for ids in labels
        )
        or labels[0] == labels[1]
    ):
        raise ValueError("A and B must be distinct single tokenizer tokens")
    return [ids[0] for ids in labels]


def encode_prefix(tokenizer, content):
    if tokenizer.padding_side != "right":
        raise ValueError("right padding required")
    ids = tokenizer.apply_chat_template(
        messages_for_pair(content), tokenize=True, add_generation_prompt=True
    )
    if (
        type(ids) is not list
        or not 1 <= len(ids) <= MAX_PREFIX_TOKENS
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError(
            "prefix outside explicit384-token bound; no truncation allowed"
        )
    return ids


def _contents(values, maximum=10000):
    if (
        type(values) is not list
        or not 1 <= len(values) <= maximum
        or len(set(values)) != len(values)
    ):
        raise ValueError("bounded unique canonical content array required")
    for content in values:
        messages_for_pair(content)
    return values


def validate_training(value):
    if type(value) is not dict or set(value) != {"training_pairs", "dev_contents"}:
        raise ValueError(
            "exact trainingpairs/devcontents fields required; no final data"
        )
    rows = value["training_pairs"]
    if type(rows) is not list or not 2 <= len(rows) <= 20000:
        raise ValueError("bounded trainingpairs required")
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {"prefix_content", "label"}
            or type(row["label"]) is not int
            or row["label"] not in (0, 1)
        ):
            raise ValueError("prefix and direct binary CEclass label required")
        messages_for_pair(row["prefix_content"])
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("both trainingclasses required")
    return rows, _contents(value["dev_contents"])


def validate_score(value):
    if type(value) is not dict or set(value) != {
        "contents",
        "adapter_file",
        "parity_contents",
        "expected_probabilities",
    }:
        raise ValueError("exact unlabeled score and parity fields required")
    _contents(value["contents"])
    _contents(value["parity_contents"], 4)
    expected = value["expected_probabilities"]
    if (
        type(expected) is not list
        or len(expected) != len(value["parity_contents"])
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
            for x in expected
        )
    ):
        raise ValueError("aligned finite frozen parity probabilities required")
    return value


def balanced_indices(rows, seed=SEED, steps=STEPS):
    pools = {
        label: [i for i, row in enumerate(rows) if row["label"] == label]
        for label in (0, 1)
    }
    if not all(pools.values()):
        raise ValueError("both balanced sampling pools required")
    rng = random.Random(seed)
    result = []
    for _ in range(steps):
        step = []
        for _ in range(ACCUMULATION):
            pair = [rng.choice(pools[0]), rng.choice(pools[1])]
            rng.shuffle(pair)
            step.append(pair)
        result.append(step)
    return result


def padded(prefixes, pad_token_id):
    import torch

    if (
        not prefixes
        or any(not row for row in prefixes)
        or type(pad_token_id) is not int
        or pad_token_id < 0
    ):
        raise ValueError("nonempty prefixes and pad token required")
    width = max(map(len, prefixes))
    ids = torch.full((len(prefixes), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(prefixes):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = 1
    return ids, mask


def adapter_path(root, name):
    path = Path(name)
    if type(name) is not str or path.name != name or not name.endswith(".safetensors"):
        raise ValueError("adapter must be a safetensors basename")
    target = Path(root) / name
    if (
        target.is_symlink()
        or not target.is_file()
        or target.stat().st_size > 16 * 1024 * 1024
    ):
        raise ValueError("bounded owned adapter file required")
    return target


def positive_probability(logits):
    if len(logits) != 2 or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError("finite A/Blogits required")
    delta = float(logits[1]) - float(logits[0])
    if delta >= 0:
        ratio = math.exp(-delta)
        return ratio / (1 + ratio)
    return 1 / (1 + math.exp(delta))


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def resource_guard(started):
    if peak_rss() > MAX_RSS or time.monotonic() - started > MAX_WALL:
        raise ValueError("frozen worker RSS/wall gate exceeded")


def score_contents(model, tokenizer, contents, labels, started):
    import torch

    from lora_v10 import pair_logits

    model.eval()
    records = []
    with torch.inference_mode():
        for content in contents:
            resource_guard(started)
            begun = time.monotonic()
            ids, mask = padded(
                [encode_prefix(tokenizer, content)], tokenizer.pad_token_id
            )
            logits = pair_logits(model, ids, mask, labels)
            probability = float(torch.softmax(logits, dim=-1)[0, 0].item())
            seconds = time.monotonic() - begun
            records.append(
                {
                    "prefix_content": content,
                    "positive_probability": probability,
                    "duration_seconds": seconds,
                }
            )
            resource_guard(started)
    return records


def new_adapter_directory(path):
    """Allow the owning parent to delete a separate-UID writer's checkpoints."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("one-use new adapter directory required")
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o777)
    return path


def run(stage, model_path, value, adapter_dir):
    started = time.monotonic()
    if stage == "train":
        rows, dev = validate_training(value)
    elif stage == "score":
        validate_score(value)
    else:
        raise ValueError("known train or score stage required")
    import safetensors
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_v10 import (
        adapter_state,
        binary_loss,
        install_lora,
        load_adapter,
        pair_logits,
    )

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model_path = Path(model_path)
    if not model_path.is_dir() or model_path.is_symlink():
        raise ValueError("existing pinned model directory required")
    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = label_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    install_lora(model)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != 230400:
        raise ValueError("actualmodel adapter parametercount mismatch")
    metadata = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "threads": THREADS,
        "seed": SEED,
        "dtype": "float32",
        "device": "cpu",
        "trainable_parameters": count,
        "label_token_ids": labels,
        "label_mapping": {"A": 0, "B": 1, "positive_probability_class": 0},
        "startup_seconds": time.monotonic() - load_started,
        "max_prefix_tokens": MAX_PREFIX_TOKENS,
    }
    for package in ("numpy", "tokenizers", "huggingface-hub"):
        metadata[package] = importlib.metadata.version(package)
    output = {
        "stage": stage,
        "passed": False,
        "metadata": metadata,
        "training_performed": False,
        "final_labels_read": False,
    }
    resource_guard(started)
    if stage == "train":
        adapter_dir = new_adapter_directory(adapter_dir)
        encoded = [encode_prefix(tokenizer, row["prefix_content"]) for row in rows]
        for content in dev:
            encode_prefix(tokenizer, content)
        output["checkpoints"] = {
            "base": {
                "step": 0,
                "adapter_file": None,
                "adapter_sha256": None,
                "records": score_contents(model, tokenizer, dev, labels, started),
            }
        }
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        training_started = time.monotonic()
        for step, batches in enumerate(balanced_indices(rows), 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for indices in batches:
                ids, mask = padded(
                    [encoded[i] for i in indices], tokenizer.pad_token_id
                )
                target = torch.tensor(
                    [rows[i]["label"] for i in indices], dtype=torch.long
                )
                loss = (
                    binary_loss(pair_logits(model, ids, mask, labels), target)
                    / ACCUMULATION
                )
                loss.backward()
                loss_sum += float(loss.detach().item())
            optimizer.step()
            resource_guard(started)
            if step == 16 and time.monotonic() - training_started > 600:
                raise ValueError("first16steps exceeded600seconds")
            if step % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "step": step,
                            "loss": loss_sum,
                            "training_seconds": time.monotonic() - training_started,
                            "peak_rss_bytes": peak_rss(),
                        }
                    ),
                    flush=True,
                )
            if step in (128, 256):
                path = adapter_dir / f"step{step}.safetensors"
                if path.exists():
                    raise ValueError("checkpoint must be new")
                save_file(adapter_state(model), str(path))
                output["checkpoints"][f"step{step}"] = {
                    "step": step,
                    "adapter_file": path.name,
                    "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "records": score_contents(model, tokenizer, dev, labels, started),
                }
        output.update(
            training_performed=True,
            completed_optimizer_steps=STEPS,
            microbatch=MICROBATCH,
            gradient_accumulation=ACCUMULATION,
            training_pairs=len(rows),
            training_seconds=time.monotonic() - training_started,
        )
    else:
        path = (
            adapter_path(adapter_dir, value["adapter_file"])
            if value["adapter_file"] is not None
            else None
        )
        if path is not None:
            load_adapter(model, load_file(str(path), device="cpu"))
        parity = score_contents(
            model, tokenizer, value["parity_contents"], labels, started
        )
        errors = [
            abs(row["positive_probability"] - expected)
            for row, expected in zip(parity, value["expected_probabilities"])
        ]
        if any(error > 1e-6 for error in errors):
            raise ValueError("fresh-process savedadapter probabilityparity failed")
        output.update(
            adapter_file=path.name if path else None,
            adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            if path
            else None,
            parity={
                "passed": True,
                "atol": 1e-6,
                "max_absolute_error": max(errors),
                "records": parity,
            },
            records=score_contents(
                model, tokenizer, value["contents"], labels, started
            ),
        )
    records = (
        [
            row
            for checkpoint in output["checkpoints"].values()
            for row in checkpoint["records"]
        ]
        if stage == "train"
        else output["records"]
    )
    resource_guard(started)
    output.update(
        passed=True,
        peak_rss_bytes=peak_rss(),
        wall_seconds=time.monotonic() - started,
        max_pair_seconds=max(row["duration_seconds"] for row in records),
        scoring_requests=len(records),
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "score"), required=True)
    for name in ("model", "input", "output", "adapter-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("new output path required")
    if args.input.is_symlink() or args.input.stat().st_size > 16 * 1024 * 1024:
        parser.error("bounded input file required")
    try:
        result = run(
            args.stage, args.model, json.loads(args.input.read_text()), args.adapter_dir
        )
    except Exception as error:  # noqa: BLE001 - serialize explicit failure without a fallback
        result = {
            "passed": False,
            "stage": args.stage,
            "failure_type": type(error).__name__,
            "failure": str(error)[:2000],
            "peak_rss_bytes": peak_rss(),
        }
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
