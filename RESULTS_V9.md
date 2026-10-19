# One-hour repair result — 19 October 2026

The complete one-shot repair attempt finished within the hour. It did not qualify the model. No rerun, changed threshold, rewritten reference, rules-based answer repair or qualified manifest followed.

| Dimension | Verified result |
| --- | --- |
| Execution and evidence | 156/156 scoring computations complete; worker stopped and temporary files removed |
| Performance | Maximum9.697313324seconds per scoring computation; pinned CPU runtime |
| Raw semantics |121/144 exact decisions (84.0278%);23/24 eligible targets;20/24 consistent permutation groups |
| Actual controller replay |143/144 exact effect decisions;23/24 eligible effects;zero unsupported allowed effects on these144 synthetic cases |
| Baselines on the same144 cases |Always-abstain and exact grammar96/144 raw; actual rules+controller120/144 exact effects,6/24 eligible effects,6 unsupported effects |
| Decision |Reject promotion: raw accuracy below90% and four inconsistent groups |
| Strict bridge |New guarded verifier has6 passing unit tests; actual failed evidence was rejected before starting PostgreSQL/Go; no passing runtime proof claimed |
| Budget |Prior140 plus new156 =296 reserved and completed scoring computations in the combined v8/v9 budget |
| Independence |AI-authored, previously exposed references; no independent human/customer validation |

The model still confuses one absent-target wording, quotation, informational instructions and policy-override attempts. All conditional/revoked/multiple-target views were raw-correct in this run. These findings are post-run diagnosis, not permission to tune or rescore this benchmark. The new144-view result is not directly comparable to the old54-view percentage.

The public protocol and output are in https://github.com/Gaurav711cgu/sentry-semantic-eval and run https://github.com/Gaurav711cgu/sentry-semantic-eval/actions/runs/39248573280 . Exact output SHA256:2c9ecda6886e59d746e7194f8b5310ba38565e059faa0170cea3dc779c7d3bac. Actual controller audit SHA256:475314d762e50a25f2818508925a134a5f2a6616fe29f365fe9cf00a5fb9ac78.

The143/144 combined result demonstrates conservative controller behavior on this fixture, not model accuracy or a general safety guarantee. The six unsupported effects from the existing rules baseline also disprove any universal safety claim for this controller. Current Sentry remains a backend engineering project with an experimental LLM component. Artha remains the stronger applied-AI evaluation project for the application handoff. Neither production deployment nor personal mastery is inferred.

