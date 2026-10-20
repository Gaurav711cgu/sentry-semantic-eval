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


def validate_replay_result(result, protocol):
    if (
        result.get("passed") is not True
        or result.get("completed_optimizer_steps") != protocol["optimizer_steps_max"]
        or result.get("learning_rate") != protocol["learning_rate"]
        or result.get("sampling_plan_sha256") != protocol["dataset"]["sampling_plan_sha256"]
        or type(result.get("scoring_requests")) is not int
        or result["scoring_requests"] != protocol["development_scoring_computations"]
        or any(result.get("metadata", {}).get(k) != v for k, v in protocol["runtime"].items())
    ):
        raise ValueError("exact complete replay optimizer, plan, runtime and scoring budget required")


def retain_score_stage(report, name, result, runtime):
    report.setdefault("score_stages", {})[name] = result
    if result.get("passed") is not True or result.get("parity", {}).get("passed") is not True:
        raise ValueError("retained scoring stage failed completion/parity: " + name)
    if any(result.get("metadata", {}).get(k) != v for k, v in runtime.items()):
        raise ValueError("retained scoring runtime drift")


def wait_grant(protocol):
    expected = {
        "experiment_id": protocol["experiment_id"],
        "source_head": os.environ["GITHUB_SHA"],
        "protocol_sha256": sha(ROOT / "protocol_v19.json"),
        "optimizer_steps_max": protocol["optimizer_steps_max"],
    }
    until = time.monotonic() + 360
    url = "https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/grants-v19/train.json"
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
        str(public / ("worker_v19.py" if stage == "train" else "worker_v10.py")),
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
    if destination.is_symlink() or destination.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("bounded worker result required")
    result = checked_json(destination.read_bytes())
    if code != 0:
        result["worker_exit_code"] = code
    return result


def unique_contents(cases):
    from pair_v10 import pair_inputs

    return sorted(
        {
            row["prefix_content"]
            for case in cases
            for row in pair_inputs(case["request"], [o["id"] for o in case["orders"]])
        }
    )


class DevelopmentRejected(Exception):
    pass


def admission_choice(selection):
    for item in selection["candidates"]:
        m = item["metrics"]
        if (
            m["raw_accuracy"] >= 0.9
            and m["eligible_recall"] is not None
            and m["eligible_recall"] >= 0.9
            and m["consistent_groups"] == m["permutation_groups"]
        ):
            return item
    return None


def main():
    from contrast_v19 import development, curriculum, paired_indices
    from evaluate_v10 import score_cases, select_candidate

    report = {
        "experiment": "broader-language-curriculum-causal-lm-v19",
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
        p = checked_json((ROOT / "protocol_v19.json").read_bytes())
        verify_sources(ROOT, p["source_sha256"])
        report["grant"] = wait_grant(p)
        report["protocol_sha256"] = sha(ROOT / "protocol_v19.json")
        report["model"] = p["model"]
        report["runtime_pins"] = p["runtime"]
        report["execution_status"] = "running"
        temporary = tempfile.TemporaryDirectory(prefix="sentry-trained-v19-")
        work = Path(temporary.name)
        work.chmod(0o755)
        public = work / "public"
        public.mkdir(mode=0o755)
        output = work / "output"
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        for name in ["pair_v10.py", "lora_v10.py", "worker_v10.py", "worker_v19.py", "sampling_v14.py"]:
            (public / name).write_bytes((ROOT / name).read_bytes())
            (public / name).chmod(0o444)
        (public / "initial.safetensors").write_bytes(
            (ROOT / "step8-v14.safetensors").read_bytes()
        )
        (public / "initial.safetensors").chmod(0o444)
        if sha(public / "initial.safetensors") != p["initial_adapter_sha256"]:
            raise ValueError("initial adapter identity mismatch")
        model = work / "model"
        report["model_acquisition_started"] = True
        download_model(model, p["model"]["files"])
        dev = development()
        dev_contents = unique_contents(dev)
        training_rows, training_families, _ = curriculum()
        train = run_worker(
            work,
            model,
            "train",
            {
                "training_pairs": training_rows,
                "training_families": training_families,
                "training_plan": paired_indices(training_rows, training_families),
                "dev_contents": dev_contents,
                "initial_adapter_file": "initial.safetensors",
                "initial_adapter_sha256": p["initial_adapter_sha256"],
            },
            "train",
            2500,
        )
        report["training"] = train
        validate_replay_result(train, p)
        report["base_development_reference"] = {
            "source": "fresh same-run unadapted model; no cross-run score cache",
            "physical_computations": len(dev_contents),
        }
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
        initial_selection = select_candidate(
            dev,
            {
                "initial": {
                    key: train["checkpoints"]["initial"][key]
                    for key in ["step", "records"]
                }
            },
            p["thresholds"],
        )
        admitted = admission_choice(selection)
        selection = {
            **(admitted or {"name": None, "step": None, "threshold": None}),
            "candidates": selection["candidates"],
            "selection_split": "dev",
            "base_threshold": base_selection["threshold"],
            "initial_threshold": initial_selection["threshold"],
            "admission": "raw_accuracy>=.9 and eligible_recall>=.9 before final inference",
        }
        frozen_selection = json.dumps(
            selection, sort_keys=True, allow_nan=False
        ).encode()
        report["selection"] = selection
        report["selection_sha256"] = hashlib.sha256(frozen_selection).hexdigest()
        print(
            "V19_SELECTION_FROZEN "
            + json.dumps(
                {
                    "name": selection["name"],
                    "threshold": selection["threshold"],
                    "sha256": report["selection_sha256"],
                }
            ),
            flush=True,
        )
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
        # Compare the already consumed, unchanged V17 recipe without retraining it.
        control_report = checked_json((ROOT / 'result-v17.json').read_bytes())
        control = control_report['retained_adapters']['step32']
        control_raw = base64.b64decode(control['base64'], validate=True)
        if len(control_raw) != control['bytes'] or hashlib.sha256(control_raw).hexdigest() != control['sha256']:
            raise ValueError('frozen control adapter integrity failure')
        control_path = output / 'adapters' / 'control-v17-step32.safetensors'
        control_path.write_bytes(control_raw)
        control_path.chmod(0o444)
        sentinel = control_report['training']['checkpoints']['step32']['records'][:4]
        control_scores = run_worker(work, model, 'score', {
            'contents': dev_contents, 'adapter_file': control_path.name,
            'parity_contents': [r['prefix_content'] for r in sentinel],
            'expected_probabilities': [r['positive_probability'] for r in sentinel],
        }, 'unchanged-control', 1100)
        retain_score_stage(report, 'unchanged_v17_step32', control_scores, p['runtime'])
        report['unchanged_control'] = {
            'source': 'V17 retained step32; 128-step original recipe, selected on exposed V17 development',
            'adapter_sha256': control['sha256'], 'threshold': 0.8,
            'metrics': score_cases(dev, control_scores['records'], 0.8),
        }
        report['final'] = {}
        report['final_inference_performed'] = False
        report['execution_status'] = 'complete'
        report['semantic_status'] = 'development_only_no_promotion'
        report['measured_demo_qualified'] = False
        report['development_admitted'] = selection['name'] is not None
        # Keep original and new construction subsets distinct in the retained report.
        report['development_subsets'] = {}
        for name, checkpoint in train['checkpoints'].items():
            ranked = next(c for c in selection['candidates'] if c['name'] == name)
            scores = {r['prefix_content']: r for r in checkpoint['records']}
            report['development_subsets'][name] = {}
            for subset, cases in [('existing', [c for c in dev if c['group_id'].startswith('dev-17-')]),
                                  ('new_constructions', [c for c in dev if c['group_id'].startswith('dev-19-')])]:
                contents = unique_contents(cases)
                report['development_subsets'][name][subset] = score_cases(cases,[scores[k] for k in contents],ranked['threshold'])
    except DevelopmentRejected as error:
        report.update(
            execution_status="complete",
            semantic_status="failed_development",
            measured_demo_qualified=False,
            final_inference_started=False,
            admission_rejection=str(error),
        )
    except BaseException as error:  # noqa: BLE001 - retain failures and cancellation after cleanup
        report.update(
            execution_status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error)[:1000],
        )
    finally:
        # Keep completed checkpoint bytes even when a later worker stage fails.
        if temporary is not None:
            for name, item in report.get("training", {}).get("checkpoints", {}).items():
                filename = item.get("adapter_file")
                if not filename:
                    continue
                try:
                    if Path(filename).name != filename:
                        raise ValueError("unsafe retained checkpoint name")
                    record = retained_file(
                        Path(temporary.name) / "output" / "adapters" / filename, 1100000
                    )
                    if record["sha256"] != item["adapter_sha256"]:
                        raise ValueError("retained checkpoint identity mismatch")
                    report.setdefault("retained_adapters", {})[name] = record
                except (OSError, ValueError) as error:
                    report.setdefault("artifact_retention_errors", []).append(
                        str(error)
                    )
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
        (ROOT / "result-v19.json").write_bytes(raw)
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        print("V19_REPORT_SHA256 " + hashlib.sha256(raw).hexdigest(), flush=True)
        for i in range(0, len(encoded), 3000):
            print("V19_REPORT_CHUNK " + encoded[i : i + 3000], flush=True)
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
