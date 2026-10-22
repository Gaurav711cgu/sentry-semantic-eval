# Task-native entailment candidate

V19 increased abstention while target-ranking errors stayed almost unchanged.
This experiment changes the learned representation and objective to a pretrained
sentence-entailment encoder. It does not claim that a controller regex repair,
additional reports, or a saved-weight parity check improves learned quality.

The selected public model is `cross-encoder/nli-deberta-v3-small`, pinned to
`fa2804872c3b4bd748f38c0185cc85775361e735`. Its publisher reports training on
SNLI and MultiNLI and defines the class order as contradiction, entailment,
neutral: https://huggingface.co/cross-encoder/nli-deberta-v3-small . Those benchmark
numbers are not AgentSentry results. The weights and tokenizer are hashed before
execution and downloaded only by the free public CPU runner.

We retain V19's 6,787 training pairs, exact 1,024-presentation sampling plan,
96-group development set, complete candidate union, and target-or-escalate
projection. A fixed sentence-pair hypothesis asks whether the customer wants
Order A cancelled now. Labels retain all scope, policy, multiple-target and
injection exclusions. Status is never fed to the classifier. The new objective
uses entailment versus the sum of contradiction and neutral probabilities.
Scores remain uncalibrated task scores, not real-world authorization probabilities.

Trainable parameters are rank8/alpha16 query/value LoRA in six encoder layers,
plus the existing NLI pooler and classifier: 740,355 parameters. The rest of the
pretrained encoder is frozen. CPU FP32, seed1019, AdamW0.0001, weight decay0.01,
microbatch2, accumulation4,128 steps. Checkpoints are base,32,64,128. This is a
different architecture and adaptation recipe, not an isolated proof that one
factor caused a change. Same-run base scores support a narrower before/after
comparison. No factorized intermediate labels are invented; that redesign is
deferred unless this smaller task-native change fails.

All20 checkpoint/threshold combinations are selected on development only.
Thresholds remain0.5/0.6/0.7/0.8/0.9. Admission requires raw accuracy and eligible
target recall at least90%, all permutation groups consistent, and per-draft
measured-forward sum at most10 seconds. No consumed run is restarted.

Before inference, confirmation_v22.py freezes24 new AI-authored utterances,
six snapshot permutations each,49 canonical candidate inputs. Their exact bytes
are disjoint from development and training. This establishes exact-input
separation, not independent human review or independent language distributions.
The worker receives no confirmation input until development selection is frozen;
it never receives confirmation labels. One fresh scoring process checks four
development parity sentinels at1e-6 before the49 confirmation inputs. A failed
development gate leaves confirmation unrun. Confirmation is never used to tune.

Maximum833 physical scoring computations:4*195development +4parity +49confirmation.
Maximum512 training forward/backward computations,1,024 presentations. RSS8GiB,
worker wall3000seconds, first16 training steps600seconds; parent deadlines and
isolated worker termination remain enforced. Complete/partial checkpoints and
raw score records are retained even after failure. Reports disclose execution,
semantic status, parity and cleanup separately. Controller replay, baselines,
strict serving integration and full release qualification follow actual passing
results; this experiment never automatically promotes a model.

No paid service, GPU, Docker, or pretrained-model execution on the Mac.
