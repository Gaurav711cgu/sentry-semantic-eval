"""Bounded corrective LoRA training; shared v10 inference primitives stay fixed."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

from worker_v10 import (
    balanced_indices,
    encode_prefix,
    label_ids,
    new_adapter_directory,
    padded,
    peak_rss,
    score_contents,
    validate_training,
)


def verified_initial(root, name, digest):
    if (
        type(name) is not str
        or Path(name).name != name
        or not name.endswith(".safetensors")
    ):
        raise ValueError("initial adapter basename required")
    path = Path(root) / name
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 1100000
    ):
        raise ValueError("bounded regular initial adapter required")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("initial adapter hash mismatch")
    return path


def guard(started):
    if peak_rss() > 8 * 1024**3 or time.monotonic() - started > 2400:
        raise ValueError("corrective training resource cap exceeded")


def run(model_path, value, adapter_dir, output_path):
    started = time.monotonic()
    if type(value) is not dict or set(value) != {
        "training_pairs",
        "dev_contents",
        "initial_adapter_file",
        "initial_adapter_sha256",
    }:
        raise ValueError("exact corrective training fields required")
    rows, dev = validate_training(
        {k: value[k] for k in ["training_pairs", "dev_contents"]}
    )
    initial = verified_initial(
        Path(__file__).parent,
        value["initial_adapter_file"],
        value["initial_adapter_sha256"],
    )
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_v10 import (
        adapter_state,
        binary_loss,
        install_lora,
        load_adapter,
        pair_logits,
    )

    torch.set_num_threads(4)
    torch.manual_seed(1019)
    random.seed(1019)
    torch.use_deterministic_algorithms(True)
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
    load_adapter(model, load_file(str(initial), device="cpu"))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != 230400:
        raise ValueError("adapter parameter identity mismatch")
    metadata = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "trainable_parameters": count,
        "seed": 1019,
        "dtype": "float32",
        "device": "cpu",
    }
    for name in [
        "transformers",
        "numpy",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
    ]:
        metadata[name] = importlib.metadata.version(name)
    output = {
        "passed": False,
        "stage": "corrective_train",
        "metadata": metadata,
        "initial_adapter_sha256": value["initial_adapter_sha256"],
        "checkpoints": {},
        "completed_optimizer_steps": 0,
        "independent_final_qualified": False,
    }
    adapter_dir = new_adapter_directory(adapter_dir)

    def persist():
        # Partial checkpoints and failure details survive until parent retention.
        temporary = output_path.with_suffix(".pending")
        temporary.write_text(json.dumps(output, allow_nan=False))
        temporary.replace(output_path)

    def checkpoint(name, step):
        path = adapter_dir / (name + ".safetensors")
        save_file(adapter_state(model), str(path))
        output["checkpoints"][name] = {
            "step": step,
            "adapter_file": path.name,
            "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": score_contents(model, tokenizer, dev, labels, started),
        }
        persist()

    try:
        encoded = [encode_prefix(tokenizer, row["prefix_content"]) for row in rows]
        checkpoint("initial", 0)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.0005
        )
        training_started = time.monotonic()
        for step, batches in enumerate(balanced_indices(rows, steps=128), 1):
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
                loss = binary_loss(pair_logits(model, ids, mask, labels), target) / 4
                loss.backward()
                loss_sum += float(loss.detach())
            optimizer.step()
            output["completed_optimizer_steps"] = step
            guard(started)
            if step <= 16 and time.monotonic() - training_started > 600:
                raise ValueError("corrective admission16-step deadline exceeded")
            if step % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "corrective_training_progress",
                            "step": step,
                            "loss": loss_sum,
                            "training_seconds": time.monotonic() - training_started,
                            "peak_rss_bytes": peak_rss(),
                        }
                    ),
                    flush=True,
                )
            if step in (32, 64, 128):
                checkpoint("step" + str(step), step)
        guard(started)
        output.update(
            passed=True,
            training_seconds=time.monotonic() - training_started,
            peak_rss_bytes=peak_rss(),
            scoring_requests=200,
        )
    except Exception as error:  # noqa: BLE001 - retain partial output and checkpoints
        output.update(failure_type=type(error).__name__, failure=str(error)[:2000])
    persist()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train"], required=True)
    for name in ["model", "input", "output", "adapter-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("one-use output required")
    try:
        result = run(
            args.model,
            json.loads(args.input.read_text()),
            args.adapter_dir,
            args.output,
        )
    except Exception as error:  # noqa: BLE001 - explicit retained failure
        result = {
            "passed": False,
            "failure_type": type(error).__name__,
            "failure": str(error)[:2000],
        }
        args.output.write_text(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
