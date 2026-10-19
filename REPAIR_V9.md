# One-hour semantic repair — 19 October 2026

This is a separate, prospectively frozen single-candidate synthetic repair experiment. The user requested a focused one-hour fix after rejecting further open-ended work. It does not rewrite v8's failed results, consumed budget or candidate-selection decision.

## Hypothesis and fixed change

The v8 model confused semantic cancellation intent with execution eligibility, especially an absent identifier, quoted instructions and conditional requests. Inputs, official pinned nonthinking ChatML and native final-token logits were checked; no scoring corruption was found. Candidate C changes only the compact system prompt: extract present unconditional intent, explicitly distinguish absent targets from execution policy, and contrast reports, conditions and questions. Option identities, order, full unfiltered options, user payloads, model, runtime, numerical scoring and gold references are unchanged. No answer repair, rules fallback or automatic retry is permitted.

## Prospective method and budget

One candidate is fixed before its first scoring over the existing, previously unscored144-view/24-group benchmark. This replaces another repeated development search with one prospective synthetic repair measurement; it does not satisfy or retroactively change v8's A/B selection eligibility. References are AI-authored, were publicly exposed, and were generated with controller and baseline details visible. This is not independent validation.

There are144 raw and12 distinct offline calibration computations,156 maximum. Together with all140 v8 reserved computations the cumulative ceiling remains296. One source-, protocol- and run-bound grant must be consumed before weight acquisition. Attempt1 only; no budget reuse after failure. The first four completed computations are also a resource-admission check, without extra model requests. Every forward request must complete within10seconds, peakRSS8GiB; worker wall limit1900seconds and hosted job40minutes. The same constraints remain enforced afterward. A speed failure stops the attempt. Cancellation and partial failure preserve reservation and transcripts; no rerun is authorized.

The worker receives no gold labels, private controller source or credentials, runs offline under a separate principal, and must stop with its temporary files removed. Standard free public CPU runner only, no artifacts/caches, paid API, Docker or local heavy inference. Logs retain complete bounded outputs and SHA256. Calibration is a diagnostic; raw choices alone feed the controller replay.

## Acceptance and claims

The existing numeric semantic requirements are unchanged: raw exact accuracy and eligible-effect recall at least90%, all24 groups consistent, zero unsupported effects allowed by the actual controller, combined exact effect accuracy at least the actual rules baseline, and eligible-effect recall at least five percentage points higher. Every request must meet10seconds with complete evidence and cleanup. Private model-free replay verifies controller and rules outcomes separately. A successful synthetic benchmark can support a new scoped measured demonstration; it cannot establish real customer performance, independent validation, v8 promotion eligibility or personal ownership. Product bridge and provider compatibility are separate checks after any numeric success. If this one candidate fails, record the failure rather than start another experiment within the hour.
