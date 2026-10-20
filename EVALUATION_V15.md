# Frozen saved-adapter evaluation after runtime diagnosis

V14 development selected step8/threshold0.8 (132/144 raw,24/24 eligible).
The diagnostic showed exact fresh-process parity for initialization and step8,
while the cross-run unadapted-base cache failed. The original parent discarded
the failed worker's details; old final exposure remains unknown. This failure
cannot be removed or called a successful final.

V15 performs no training and never selects different adapter weights or their
threshold. Freeze a new36-scenario/216-view AI-authored synthetic final before
inference. Compare unadapted base, v12 initialization and selected v14step8.
Base reference is recomputed on the50 exposed development prefixes on the same
runner. Its threshold alone is selected from development; initialization keeps
its old development threshold and selected adapter keeps0.8. Separate fresh
processes must match sentinel scores within1e-6 before final scoring. Exact
failure responses survive in the parent report before validation rejects them.

Maximum scoring budget:50 base reference +3*(73 final+4 parity)=281 computations.
Do not transfer unused prior reservations or rerun consumed attempts. Source,
model/runtime, adapter hashes, final packet, thresholds, raw/controller gates,
and private reader are frozen before execution. All prior failures retained.

Acceptance remains raw accuracy>=90%, actual-controller eligible recall>=90%,
all structural groups consistent, no observed unsupported controller effects,
combined accuracy>=rules and recall>=rules+5percentagepoints, latency<=10seconds
per uncached draft estimate, complete outputs, parity and cleanup. Independent
human/customer qualification remains false. No paid compute, Docker or heavy
local model inference is required or authorized by this evaluation.
