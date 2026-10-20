# V17: broader-language training did not improve raw development accuracy

All128 optimizer steps and1015 development scoring computations completed.
The parent then failed because new case IDs used dev17-* while the selector
requires a separator after dev. Preserve this implementation failure. A tested,
separate identity-only recovery maps IDs to dev-17-* and recomputes stored scores
without changing model inputs, references, probabilities, thresholds or weights.
No retraining or new scoring occurred during recovery.

On the same72-scenario/432-correlated-view development bank, initialization
scores360/432 (83.3%) with54/72 eligible targets (75%). The highest-ranked trained
checkpoint, step32 at threshold0.8, also scores360/432 (83.3%), with60/72 eligible
targets (83.3%). No checkpoint passes both90% admission gates. Final inference
never started; no qualified manifest or promotion follows.

Retain every adapter, score, failed execution, recovery and consumed budget.
These results do not show an overall generalization gain. More training is not
guaranteed to cross90%; a later study needs a materially different, prospectively
frozen hypothesis and evaluation rather than threshold tuning or blind repeats.
References remain AI-authored synthetic; independent human/customer qualification
is not claimed. Application tests remain186 local passes, separate from AI quality.
