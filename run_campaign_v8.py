"""One consumed development attempt on free standard public CI; no final claim."""

import argparse
import base64
import gzip
import hashlib
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from contracts import (
    validate_grant,
    verify_model,
    worker_error_tail,
)
from contracts_v7 import validate_events
from campaign_v8 import (
    STAGE_CALLS,
    summarize,
    validate_ledger,
    verify_result,
    select_candidate,
    finish,
)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(STAGE_CALLS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT / ("result-v8-" + args.stage + ".json")
    if args.output.exists() or args.output.is_symlink():
        parser.error("one-use result path required")
    report = {
        "passed": False,
        "independent_final_qualified": False,
        "scope": "Measured AI demo campaign; no independent-validation claim",
        "execution_status": "not_started",
        "performance_status": "not_evaluated",
        "semantic_status": "not_evaluated",
        "evidence_status": "unverified",
        "independent_validation_status": "not_claimed",
        "stage": args.stage,
        "calls_consumed": 0,
        "completed_calls": 0,
        "model_acquisition_started": False,
        "events": [],
    }
    process = None
    temporary = None
    removed = False
    graded = False
    complete = False
    failure = None
    phase = "admission"
    try:
        attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "0"))
        if attempt != 1:
            raise ValueError("attempt1 only; no reused model budgets")
        if os.environ.get("GITHUB_REPOSITORY") != "Gaurav711cgu/sentry-semantic-eval":
            raise ValueError("wrong repository")
        run_id = int(os.environ["GITHUB_RUN_ID"])
        head = os.environ["GITHUB_SHA"]
        protocol_path = ROOT / ("protocol_v8_" + args.stage + ".json")
        p = json.loads(protocol_path.read_text())
        if p["calls"] != STAGE_CALLS[args.stage] or len(p["requests"]) != p["calls"]:
            raise ValueError("exact stage computation budget required")
        for name, wanted in p["source_sha256"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or (ROOT / path).is_symlink():
                raise ValueError("unsafe source binding")
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != wanted:
                raise ValueError("source drift")
        expected = {
            "experiment_id": p["experiment_id"],
            "source_head": head,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "calls": p["calls"],
            "campaign_id": "measured-v8-20261019",
            "stage": args.stage,
        }
        base_url = (
            "https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/"
        )
        grant_url = base_url + "grants-v8/" + args.stage + ".json"
        deadline = time.monotonic() + 360
        while True:
            try:
                with urlopen(grant_url, timeout=15) as response:
                    grant = json.loads(response.read(65536))
                validate_grant(grant, expected, run_id, attempt)
                break
            except HTTPError as error:
                if error.code != 404:
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError("no consumed grant")
                time.sleep(5)
        report.update(
            grant=grant,
            calls_consumed=p["calls"],
            source_sha256=p["source_sha256"],
            model=p["model"],
            protocol_sha256=expected["protocol_sha256"],
        )

        def fetch_json(relative):
            with urlopen(base_url + relative, timeout=30) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError("bounded controller record required")
            return json.loads(payload)

        ledger = fetch_json("campaign-v8-ledger.json")
        report["campaign_calls_reserved"] = validate_ledger(ledger)
        if ledger["grants"].get(args.stage) != grant:
            raise ValueError("consumed grant missing from campaign ledger")
        if args.stage != "profile":
            profile_p = json.loads((ROOT / "protocol_v8_profile.json").read_text())
            admission = fetch_json("results-v8/profile.json")
            verified = verify_result(admission, profile_p, ROOT)
            admitted = [
                name
                for name, value in verified["performance_by_candidate"].items()
                if value["passed"]
            ]
            if args.stage[-1] not in admitted:
                raise ValueError("candidate failed admission")
            if args.stage.startswith("confirm-"):
                developments = {}
                for candidate in admitted:
                    candidate_p = json.loads(
                        (ROOT / ("protocol_v8_dev-" + candidate + ".json")).read_text()
                    )
                    development = fetch_json("results-v8/dev-" + candidate + ".json")
                    verify_result(development, candidate_p, ROOT)
                    developments[candidate] = development
                selected = select_candidate(developments)
                if selected != args.stage[-1]:
                    raise ValueError(
                        "confirmatory candidate differs from deterministic selection"
                    )
        report["execution_status"] = "running"
        temporary = tempfile.TemporaryDirectory(prefix="sentry-measured-v8-")
        work = Path(temporary.name)
        work.chmod(0o755)
        model = work / "model"
        model.mkdir(mode=0o755)
        phase = "acquisition"
        report["model_acquisition_started"] = True
        for name, record in p["model"]["files"].items():
            if Path(name).name != name:
                raise ValueError("unsafe model filename")
            url = record["url"]
            size = 0
            digest = hashlib.sha256()
            with (
                urlopen(url, timeout=60) as response,
                (model / name).open("xb") as output,
            ):
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > record["bytes"]:
                        raise ValueError("model byte bound exceeded")
                    digest.update(block)
                    output.write(block)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("model acquisition mismatch")
            (model / name).chmod(0o444)
        verify_model(model, p["model"]["files"])
        public = work / "public"
        public.mkdir(mode=0o755)
        for name in ["worker_v7.py", "contracts.py", "prompts_v7.py"]:
            shutil.copyfile(ROOT / name, public / name)
            (public / name).chmod(0o444)
        shutil.copyfile(protocol_path, public / "protocol.json")
        (public / "protocol.json").chmod(0o444)
        private = work / "private"
        private.mkdir(mode=0o700)
        canary = private / "labels-canary"
        canary.write_bytes(b"invented-private-canary")
        # No repository access, labels, credential environment or live networking.
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
            str(public / "worker_v7.py"),
            "--protocol",
            str(public / "protocol.json"),
            "--model",
            str(model),
            "--canary",
            str(canary),
        ]
        phase = "inference"
        begun = time.monotonic()
        overall = begun + p["max_worker_wall_seconds"]
        next_deadline = min(overall, begun + p["max_load_seconds"])
        buffer = b""
        bytes_read = 0
        with (work / "worker.stderr").open("w+b") as errors:
            process = subprocess.Popen(
                command,
                cwd=public,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=errors,
                start_new_session=True,
                close_fds=True,
            )
            finished = False
            while not finished:
                remaining = next_deadline - time.monotonic()
                if (
                    remaining <= 0
                    or not select.select([process.stdout], [], [], remaining)[0]
                ):
                    raise TimeoutError("frozen worker deadline")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("worker ended without finished event")
                buffer += chunk
                bytes_read += len(chunk)
                if len(buffer) > 100000 or bytes_read > 2 * 1024 * 1024:
                    raise ValueError("transcript byte bound exceeded")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    event = json.loads(line)
                    report["events"].append(event)
                    if len(report["events"]) > 2 * p["calls"] + 2:
                        raise ValueError("excess transcript events")
                    if event.get("event") in ["started", "loaded"]:
                        next_deadline = min(
                            overall, time.monotonic() + p["max_forward_seconds"]
                        )
                    if event.get("event") == "completed":
                        report["completed_calls"] += 1
                    if event.get("event") == "finished":
                        finished = True
            if process.wait(timeout=10) != 0:
                raise RuntimeError("worker failed after transcript")
            if buffer.strip() or process.stdout.read(100001).strip():
                raise ValueError("trailing worker output")
        records = validate_events(report["events"], p)
        complete = True
        phase = "profile_validation"
        report.update(summarize(records, p, ROOT))
        graded = (
            True  # Stage-specific grading remains separate from independent validation.
        )
        report["worker_wall_seconds"] = time.monotonic() - begun
    except BaseException as error:  # noqa: BLE001 - cancellation must still record cleanup and failure
        failure = type(error).__name__
        report.update(
            failure_type=failure, failure_phase=phase, failure_message=str(error)[:300]
        )
        if temporary is not None and (Path(temporary.name) / "worker.stderr").is_file():
            report["worker_error_tail"] = worker_error_tail(
                Path(temporary.name) / "worker.stderr"
            )
    finally:
        if process is not None and process.poll() is None:
            for signal in ["-TERM", "-KILL"]:
                if process.poll() is not None:
                    break
                subprocess.run(
                    ["sudo", "-n", "kill", signal, "--", "-" + str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        stopped = process is None or process.poll() is not None
        try:
            if temporary is not None:
                temporary.cleanup()
            removed = temporary is None or not Path(temporary.name).exists()
        except OSError as error:
            failure = failure or type(error).__name__
            report["cleanup_failure_type"] = type(error).__name__
        report.update(owned_worker_stopped=stopped, temporary_artifacts_removed=removed)
        report.update(
            finish(
                report,
                complete=complete,
                graded=graded,
                stopped=stopped,
                removed=removed,
                failure=failure,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, allow_nan=False).encode()
        args.output.write_bytes(payload)
        # Logs and job summaries use no billed artifact/cache storage.
        encoded = base64.b64encode(gzip.compress(payload)).decode()
        print("EVAL_REPORT_SHA256 " + hashlib.sha256(payload).hexdigest())
        for i in range(0, len(encoded), 3000):
            print("EVAL_REPORT_CHUNK " + encoded[i : i + 3000])
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in [
                        "passed",
                        "completed_calls",
                        "performance_status",
                        "semantic_status",
                        "independent_final_qualified",
                        "failure_type",
                        "failure_phase",
                    ]
                }
            )
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
