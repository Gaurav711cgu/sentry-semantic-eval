# V15: model runtime repaired; frozen semantics still fail

No retraining occurred. The selected v14 step8 adapter and threshold0.8 were
frozen before the fresh AI-authored final. The evaluation completed281 scoring
computations:50 fresh base-development reference computations plus three controls
with4 parity and73 final-prefix computations each. All three controls passed
fresh-process parity with maximum absolute error0.0 and completed cleanup.

The original cross-run unadapted-base cache differs from this runner by at most
0.0000037551. That exceeded the unchanged1e-6 parity threshold. The new evaluation
recomputed its base control rather than loosening parity or retraining the model.
Exact hardware/kernel cause of the small cross-run difference is not established.

| Control | Raw exact views | Eligible raw hits |
| --- | --- | --- |
| Unadapted base |162/216 (75.0%)|0/36|
| V12 initialization |186/216 (86.1%)|24/36|
| Development-selected v14 step8 |174/216 (80.6%)|24/36|

The selected checkpoint does not beat initialization on this final. The final
contains36 scenarios with six correlated structural views each. Permutation
consistency is engineered, not learned. Selected uncached draft-forward estimate
peaks at1.10655seconds; this is not complete deployed end-to-end latency.

The actual frozen private controller gives192/216 correct effect decisions,
24/36 eligible effects and12 unsupported views across two scenarios: a quoted
instruction and a request to invent a policy citation. No real effects executed.
Original failed gates: raw accuracy, eligible-effect recall and zero unsupported
effects. These failures remain frozen and reproducible; no promotion occurred.

After seeing the final, focused controller regressions exposed and repaired those
two veto gaps. An explicitly exposed replay then gives204/216 correct effect
decisions and zero unsupported views, with eligible recall still24/36. This is
post-final regression evidence, not a new final pass or learned-model gain.
The original audit reproduces byte-for-byte from archived private source.

The local application suite passes186 tests after controller hardening, removal
of reverse DNS from simulated-recipient startup and a bounded64-connection pending
queue for concurrent retries. Hosted application CI does not cover these new
local changes. Public hosted model tests and score evidence are separate.

Keep the adapter experimental. Do not tune from this final, label it independent,
claim real customer performance or mint a qualified manifest. Retained reports,
protocols and weights support a measured failed generalization study. A later
training change requires a new explicit protocol, budget and untouched evaluation.
