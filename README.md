# AgentSentry learned intent model and evaluation lab

This student project studies whether a small supervised language model can
identify current cancellation authorization while a separate backend governs
action execution. It is not a customer deployment or independent validation.
The public repository contains original code, AI-authored synthetic scenarios,
protocols, retained adapters and scored outputs; no private backend code,
credentials or customer data are published. All evaluation sources are exposed.

## Current measured state

The learned component freezes SmolLM2-135M-Instruct and trains 230,400 rank-four
q/v LoRA parameters using binary next-token classification. It scores every
listed or mentioned order without eligibility filtering, then escalates unless
exactly one candidate passes the development-selected threshold. A separate
private controller validates proposed effects. No rules fallback repairs targets.

| Experiment | Exposed development raw exact | Eligible-target recall | Final status |
| --- | --- | --- | --- |
| V10 trained adapters | 96/144 (66.7%) | 12/24 (50%) | Unadapted base was selected; trained adapter not promoted |
| V11 corrective curriculum | 126/144 (87.5%) | 18/24 (75%) | Admission failed; no final inference |
| V12 context curriculum | 126/144 (87.5%) | 24/24 (100%) | Admission failed; no final inference |
| V13 context and condition curriculum | Initialization still ranks best | No improvement over V12 | Admission failed; no final inference |

These are 24 scenarios, each under six correlated snapshot permutations.
Permutation invariance is structural normalization, not learned robustness.
V12 still misses two absent-order requests and wrongly accepts one conditional
request. Higher eligible recall does not establish better overall accuracy.
No trained adapter currently has measured-demo qualification. V13 completed
without improving on its initialization; its new checkpoints also failed
admission. The reserved final remains untouched.

Protocols and results: [V10](RESULTS_V10.md), [V11](RESULTS_V11.md),
[V12](RESULTS_V12.md), [V13 result](RESULTS_V13.md),
[V13 frozen plan](TRAINING_V13.md).
The model architecture, normalization, train/serve scoring primitives and failed
history remain visible. The original Qwen prompt experiments below use different
banks; their percentages are not directly comparable to this development table.

## Recompute the latest completed raw result without a model

Use Python standard library only, from this repository. No download, API key,
model inference, Docker or paid compute is needed:

```sh
python3 - <<'PYCODE'
import json
from data_v12 import build_split
from evaluate_v10 import score_cases
report = json.load(open("result-v12.json"))
checkpoint = report["training"]["checkpoints"]["step128"]
metrics = score_cases(build_split("dev"), checkpoint["records"], 0.5)
print({key: metrics[key] for key in
       ["raw_exact_correct", "cases", "eligible_raw_target_hits", "eligible_targets"]})
assert metrics["raw_exact_correct"] == 126
assert metrics["eligible_raw_target_hits"] == 24
PYCODE
```

This recomputes raw predictions from retained scores. It does not re-execute the
model or impersonate private-controller/runtime proof. A new training run needs
its exact prospectively consumed grant; rerunning a consumed attempt is rejected.
All heavy pretrained-model work runs on standard public Linux CPU runners.

## Historical prompt-only experiments


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

## Measured demo campaign v8

Two compact prompts are frozen with the existing pinned Qwen3-4B Q4_K_M model/runtime. `campaign_v8.json` fixes a296-request maximum:8 admission requests,66 development computations per admitted candidate and156 confirmatory computations for at most one selected candidate. Every stage has an exact source/protocol/run-bound consumed grant. The mutable campaign ledger counts all reserved calls before execution and rejects duplicate runs or both confirmatory candidates.

Exact content-free input deduplication reduces the old108 logical requests to66 physical computations (54 raw,12 distinct content-free); every logical identity retains its physical computation/cache binding. Raw requests are never deduplicated. Cache identities include weights/revision/runtime, rendered messages, option order and scoring configuration. Calibrated scores are diagnostics only; the proposed provider uses one raw scoring request per draft.

The144-view confirmatory packet and references were authored before v8 candidate inference, with24 groups/six permutations and12 challenge types. They are AI-authored synthetic scenarios. Their author saw the controller, baselines and earlier failures; they are not independent human validation, real customer traces or proof of pretraining/family independence. Original review disagreements and prior failures remain preserved. Model status cannot be promoted from this lab alone: the actual private controller/rules comparison and strict approval bridge are separate mandatory checks.

The manual `campaign-v8.yml` workflow admits candidates from actual retained profiles and enforces deterministic development selection before a confirmatory run. Frozen10-second scoring and8GiB RSS gates are unchanged. Reports separate execution, performance, raw semantics, evidence integrity and independent validation. No artifact/cache uploads, paid hardware, local LLM inference or service auto-submission are configured.

The frozen v8 admission run39237198966 completed8/8 requests: candidate A max4.8053 seconds, B max5.9193 seconds, peak RSS4,594,176,000 bytes, owned stop/cleanup passed. This is runtime admission only. Review then found prerequisite reports needed explicit protocol/model/run/grant/runtime binding and terminal-failed candidates needed exclusion rather than blocking a passing peer. Separately versioned guarded sources and workflow repair those checks before any development inference; consumed profile sources remain unchanged. Future stage protocols pin exact prerequisite report bytes and consumed grants. Seven guard regressions plus19 existing tests pass. No extra admission calls are consumed by this repair.

Both guarded development runs completed66/66 computations and passed speed/cleanup, but failed semantics. A run39238375635 achieved29/54 raw exact,11/12 eligible raw targets and8/9 consistent groups, maximum7.9724 seconds. B run39238380532 achieved36/54,11/12 and7/9, maximum6.8811 seconds. The actual private controller replay produced53/54 exact allowed-effect decisions and11/12 eligible targets with zero unsupported allowed effects on this fixture for each; those filtered outcomes do not repair raw model errors. Exact grammar scored54/54 raw on these narrow templates. No candidate was selected. All140 reserved computations completed; confirmatory reservation/execution stays0. The144-view confirmatory packet remains unscored, thresholds unchanged and no qualified provider manifest minted. Results and explicit rejection are retained under results-v8/ and selection-v8.json.
