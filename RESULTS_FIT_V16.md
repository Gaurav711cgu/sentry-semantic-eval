# V16: sampled training fit is high; held-out generalization remains failed

The fixed training-corpus probe completed216 physical computations with no
training, threshold selection or final inference. It samples104 canonical pairs
by class, semantic family and presentation during the first eight v14 steps.
That membership is not a claim that a warm-start model never saw the content
in any earlier training phase. Labels and family metadata never enter the model.

| Frozen adapter | Frozen threshold | Correct sampled training pairs |
| --- | --- | --- |
| V12 initialization |0.5|102/104 (98.1%)|
| V14 development-selected step8 |0.8|101/104 (97.1%)|

Both models passed fresh-process parity with zero difference. Selected errors
concern a missing-screen cancellation, preserving a different order, and a
policy-override request. The probe is stratified rather than representative;
104 samples do not establish full-corpus fit, capacity sufficiency or statistical
superiority of one adapter. Comparing different frozen thresholds is a decision
comparison, not a calibrated-probability comparison.

Alongside the consumed v15 final (80.6% selected raw accuracy), high sampled
training fit points toward limited language generalization and weak development
selection coverage. It does not establish a single causal mechanism. Repeating
training or increasing paid hardware alone is not justified by this diagnostic.

Next work should broaden utterance diversity and development scenarios, group
paraphrases by declared source before splitting, preserve all consumed finals,
and freeze a new training/selection study before inference. A new split guard
rejects exact canonical-prefix leakage even after identifier renaming, conflicting
permutation references and declared utterance families crossing partitions. It
cannot establish that differently named sources are independently authored.

No model promotion, independent human/customer qualification or live effects
occurred. Runtime repair and186 local application tests remain prior evidence;
the learned-model qualification gate is still failed.
