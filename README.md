# Sentry semantic development comparison

This standalone lab publishes only original evaluation harness code and synthetic, already exposed development inputs. It contains no private backend implementation, credentials, production data or final holdout. The model is an attributed dependency, not our implementation. The first experiment isolates model identity: Qwen2.5-1.5B-Instruct replaces the earlier0.5B model with the same public54 cases, option ordering, prompts, letter scoring and raw/content-free calculation. Passing this diagnostic is not independent AI qualification.

## Custody and finite budget

Manual workflow only. A trusted controller commits an exact run/source/protocol-bound consumed grant before weight acquisition; attempt1 only and all108 calls are consumed even on failure. The inference process runs under Linux UID65534 in a separate network/PID namespace, with a denied parent-private canary and no inherited credentials. The parent retains grading labels. These synthetic labels are public development references, not an unseen final. A repository administrator can bypass these controls.

Model identity, runtime, sources, forward/resource limits and request messages are frozen in protocol.json before dispatch. The worker returns scores only. It has no recipient or action API. Complete ordered transcripts and cleanup are required for run completion. Raw semantic gates and independent final qualification are separately reported. The real Sentry controller is evaluated later from retained predictions in the private product workspace; this lab does not impersonate that boundary.

## Costs and outputs

Use standard ubuntu-24.04 runners in this public repository. Do not switch to paid larger runners. No Actions caches or artifact uploads are configured. Bounded report bytes are compressed and split into job-log lines with a SHA-256 for later integrity checking. Model files live in a disposable directory and are removed after the owned process stops. No model weights are acquired or inferred on the owner's Mac.

`python -m unittest -v` tests admission, exact model files, transcript completeness, resource bounds and grading. It never loads the model. `python run_probe.py` is for the protected manual CI job only. A failed quality gate remains failed; do not change consumed protocol/source, rerun a consumed job, fabricate human reviews, or declare independent final qualification from a public template set.

## BF16 configuration experiment v6

The immutable FP32 v5 attempt (run39162919569) consumed108 calls and completed0 forwards. Its load resource check failed before emitting load metrics, so the exact breached measurement is unknown. Worker stop and temporary cleanup succeeded; there is no semantic result.

The separately frozen v6 protocol uses the same model,108 requests and quality/resource limits, but BF16 numerical precision. This is a configuration comparison, not a model-size-only ablation. It emits load measurements before enforcing the unchanged load limit, and records the unconstrained vocabulary top token from each existing forward. Letter scores remain unchanged; diagnostics cannot certify instruction following if the top token is not an allowed letter. No extra generation or inference calls are reserved. v6 has a distinct run-bound consumed grant, protocol and executable files; v5 remains untouched. Independent final qualification remains false.

## Quantized CPU admission v7

v6 run39163388530 timed out after51/108 completed scoring calls. Its complete eligible raw subset already failed the90% recall gate (10/12); remaining calls cannot repair that gate. The consumed v5/v6 files remain immutable.

v7 reserves only four scoring requests to admit a different CPU configuration: Qwen3-4B-Instruct-2507, publisher Unsloth Q4_K_M GGUF, llama-cpp-python0.3.36. The model, quantization and inference backend change together. Two longest rendered exposed inputs per mode are fixed before execution. All four must complete, each within10 seconds, with complete lifecycle/resource/isolation evidence and cleanup. The process also enforces a60-second individual hard timeout and8GiB peak RSS. Passing admits further evaluation; it does not grade semantic quality. No full-suite calls are authorized by its four-call grant.

The exact official Qwen3 template is checked on all108 exposed messages. The pinned llama-cpp-python version does not fill its Python score matrix when logits_all=False; this worker reads the requested final-token logits directly from the native context. It resets context between requests, checks finite full-vocabulary logits, records the top token and performs no generation. Semantic scoring-request counts do not claim single low-level kernel or microbatch calls. Model/card/license revisions and CPU wheel digest are pinned before dispatch.
