"""Run one prospectively consumed, CPU-only supervised LM adaptation experiment.

Training and selection see development inputs; final inference begins only
after the parent freezes the selected checkpoint and threshold. The worker has
no repository, credential, final-label or network access. Earlier experiments
and their budgets are never reused.
"""

import base64
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
OWNED_PROCESSES = []


def checked_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root, bindings):
    if not bindings:
        raise ValueError("nonempty exact source bindings required")
    for name, digest in bindings.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
            raise ValueError("unsafe source path")
        if sha(root / path) != digest:
            raise ValueError("frozen source drift: " + name)


def validate_training_grant(grant, expected, run_id, attempt):
    if (
        type(grant) is not dict
        or set(grant) != set(expected) | {"run_id", "attempt", "status"}
        or type(run_id) is not int
        or run_id <= 0
        or type(attempt) is not int
        or attempt != 1
        or type(grant.get("run_id")) is not int
        or grant["run_id"] != run_id
        or type(grant.get("attempt")) is not int
        or grant["attempt"] != 1
        or grant.get("status") != "consumed_before_execution"
        or any(
            type(grant[k]) is not type(v) or grant[k] != v for k, v in expected.items()
        )
    ):
        raise ValueError("exact consumed run/source/protocol/budget grant required")


def retained_file(path, maximum):
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= maximum
    ):
        raise ValueError("bounded regular retained file required")
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "base64": base64.b64encode(data).decode("ascii"),
    }


def wait_grant(protocol):
    expected = {
        "experiment_id": protocol["experiment_id"],
        "source_head": os.environ["GITHUB_SHA"],
        "protocol_sha256": sha(ROOT / "protocol_v10.json"),
        "optimizer_steps_max": protocol["optimizer_steps_max"],
    }
    until = time.monotonic() + 360
    url = "https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/grants-v10/train.json"
    while True:
        try:
            with urlopen(url, timeout=15) as response:
                grant = checked_json(response.read(65537))
            validate_training_grant(
                grant,
                expected,
                int(os.environ["GITHUB_RUN_ID"]),
                int(os.environ["GITHUB_RUN_ATTEMPT"]),
            )
            return grant
        except HTTPError as error:
            if error.code != 404 or time.monotonic() >= until:
                raise
            time.sleep(5)


def download_model(model, records):
    model.mkdir(mode=0o755)
    for name, record in records.items():
        if Path(name).name != name:
            raise ValueError("unsafe model filename")
        size = 0
        digest = hashlib.sha256()
        with (
            urlopen(record["url"], timeout=60) as response,
            (model / name).open("xb") as output,
        ):
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > record["bytes"]:
                    raise ValueError("model byte bound exceeded")
                digest.update(block)
                output.write(block)
        if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("model byte/hash identity mismatch")
        (model / name).chmod(0o444)


def run_worker(work, model, stage, payload, output_name, timeout):
    """A fresh UID/network-isolated process; deadlines also enforced by parent."""
    public = work / "public"
    output = work / "output"
    input_path = public / (output_name + ".input.json")
    input_path.write_text(json.dumps(payload, allow_nan=False))
    input_path.chmod(0o444)
    destination = output / (output_name + ".json")
    command = [
        "sudo",
        "-n",
        "unshare",
        "--net",
        "--pid",
        "--fork",
        "--mount-proc",
        "--kill-child",
        "setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--no-new-privs",
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        "HOME=/tmp",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        os.environ["PROBE_PYTHON"],
        str(public / "worker_v10.py"),
        "--stage",
        stage,
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output",
        str(destination),
        "--adapter-dir",
        str(output / "adapters"),
    ]
    process = subprocess.Popen(
        command, cwd=public, stdin=subprocess.DEVNULL, start_new_session=True
    )
    OWNED_PROCESSES.append(process)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        subprocess.run(
            ["sudo", "-n", "kill", "-TERM", "--", "-" + str(process.pid)], check=False
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["sudo", "-n", "kill", "-KILL", "--", "-" + str(process.pid)],
                check=False,
            )
            process.wait(timeout=10)
        raise
    if code != 0:
        raise RuntimeError("isolated worker failed: " + str(code))
    if destination.is_symlink() or destination.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("bounded worker result required")
    return checked_json(destination.read_bytes())


def unique_contents(cases):
    from pair_v10 import pair_inputs

    return sorted(
        {
            row["prefix_content"]
            for case in cases
            for row in pair_inputs(case["request"], [o["id"] for o in case["orders"]])
        }
    )


def main():
    from data_v10 import build_split, training_pairs
    from evaluate_v10 import score_cases, select_candidate

    report = {
        "experiment": "supervised-causal-lm-v10",
        "execution_status": "not_started",
        "semantic_status": "not_evaluated",
        "independent_final_qualified": False,
        "independent_validation_status": "not_claimed",
        "references": "AI-authored synthetic",
        "old_v8_v9_budgets_reused": False,
        "model_acquisition_started": False,
    }
    temporary = None
    try:
        if os.environ.get("GITHUB_REPOSITORY") != "Gaurav711cgu/sentry-semantic-eval":
            raise ValueError("public zero-cost runner repository required")
        p = checked_json((ROOT / "protocol_v10.json").read_bytes())
        verify_sources(ROOT, p["source_sha256"])
        report["grant"] = wait_grant(p)
        report["protocol_sha256"] = sha(ROOT / "protocol_v10.json")
        report["model"] = p["model"]
        report["runtime_pins"] = p["runtime"]
        report["execution_status"] = "running"
        temporary = tempfile.TemporaryDirectory(prefix="sentry-trained-v10-")
        work = Path(temporary.name)
        work.chmod(0o755)
        public = work / "public"
        public.mkdir(mode=0o755)
        output = work / "output"
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        for name in ["pair_v10.py", "lora_v10.py", "worker_v10.py"]:
            (public / name).write_bytes((ROOT / name).read_bytes())
            (public / name).chmod(0o444)
        model = work / "model"
        report["model_acquisition_started"] = True
        download_model(model, p["model"]["files"])
        dev = build_split("dev")
        final = build_split("final")
        dev_contents = unique_contents(dev)
        final_contents = unique_contents(final)
        train = run_worker(
            work,
            model,
            "train",
            {"training_pairs": training_pairs(), "dev_contents": dev_contents},
            "train",
            5550,
        )
        report["training"] = train
        if (
            not train.get("passed")
            or train.get("completed_optimizer_steps") != p["optimizer_steps_max"]
        ):
            raise ValueError("complete bounded training transcript required")
        if any(
            train["metadata"].get(name) != version
            for name, version in p["runtime"].items()
        ):
            raise ValueError("training runtime identity mismatch")
        # This manifest is created before any final input is sent to a worker.
        selection = select_candidate(
            dev,
            {
                name: {key: checkpoint[key] for key in ["step", "records"]}
                for name, checkpoint in train["checkpoints"].items()
            },
            p["thresholds"],
        )
        base_selection = select_candidate(
            dev,
            {
                "base": {
                    key: train["checkpoints"]["base"][key]
                    for key in ["step", "records"]
                }
            },
            p["thresholds"],
        )
        selection["base_threshold"] = base_selection["threshold"]
        frozen_selection = json.dumps(
            selection, sort_keys=True, allow_nan=False
        ).encode()
        report["selection"] = selection
        report["selection_sha256"] = hashlib.sha256(frozen_selection).hexdigest()
        print("V10_SELECTION_FROZEN " + frozen_selection.decode(), flush=True)
        report["retained_adapters"] = {}
        for name, checkpoint in train["checkpoints"].items():
            if checkpoint.get("adapter_file"):
                filename = checkpoint["adapter_file"]
                if Path(filename).name != filename:
                    raise ValueError("unsafe adapter filename")
                record = retained_file(output / "adapters" / filename, 1100000)
                if record["sha256"] != checkpoint["adapter_sha256"]:
                    raise ValueError("checkpoint adapter hash mismatch")
                report["retained_adapters"][name] = record
        results = {}
        selected_name = selection["name"]
        for name in dict.fromkeys(["base", selected_name]):
            checkpoint = train["checkpoints"][name]
            sentinel = checkpoint["records"][:4]
            payload = {
                "contents": final_contents,
                "adapter_file": checkpoint.get("adapter_file"),
                "parity_contents": [r["prefix_content"] for r in sentinel],
                "expected_probabilities": [r["positive_probability"] for r in sentinel],
            }
            result = run_worker(work, model, "score", payload, "final-" + name, 1100)
            if not result.get("passed") or not result.get("parity", {}).get("passed"):
                raise ValueError(
                    "fresh-process serving parity required before final grading"
                )
            if any(
                result["metadata"].get(key) != value
                for key, value in p["runtime"].items()
            ):
                raise ValueError("serving runtime identity mismatch")
            threshold = (
                selection["threshold"]
                if name == selected_name
                else selection["base_threshold"]
            )
            result["metrics"] = score_cases(final, result["records"], threshold)
            results[name] = result
        report["final"] = results
        candidate = results[selected_name]["metrics"]
        report["execution_status"] = "complete"
        report["semantic_status"] = "pending_actual_controller_audit"
        report["public_raw_gate_passed"] = (
            candidate["raw_accuracy"] >= 0.9
            and candidate["eligible_recall"] >= 0.9
            and candidate["consistent_groups"] == candidate["permutation_groups"]
        )
        report["measured_demo_qualified"] = False
    except BaseException as error:  # noqa: BLE001 - retain failures and cancellation after cleanup
        report.update(
            execution_status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error)[:1000],
        )
    finally:
        report["owned_worker_stopped"] = all(
            process.poll() is not None for process in OWNED_PROCESSES
        )
        removed = temporary is None
        if temporary is not None:
            try:
                temporary.cleanup()
                removed = not Path(temporary.name).exists()
            except OSError as error:
                report["cleanup_failure"] = str(error)[:300]
        report["temporary_artifacts_removed"] = removed
        raw = json.dumps(report, indent=2, allow_nan=False).encode()
        (ROOT / "result-v10.json").write_bytes(raw)
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        print("V10_REPORT_SHA256 " + hashlib.sha256(raw).hexdigest(), flush=True)
        for i in range(0, len(encoded), 3000):
            print("V10_REPORT_CHUNK " + encoded[i : i + 3000], flush=True)
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in [
                        "execution_status",
                        "semantic_status",
                        "public_raw_gate_passed",
                        "failure_type",
                        "failure_message",
                    ]
                }
            ),
            flush=True,
        )
    return 0 if report["execution_status"] == "complete" and removed else 1


if __name__ == "__main__":
    raise SystemExit(main())
