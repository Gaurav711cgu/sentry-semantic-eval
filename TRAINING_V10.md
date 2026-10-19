# Supervised causal-language-model adaptation

This is a new experiment, not a rerun or a reclassification of failed v8/v9
results. It adapts the real SmolLM2-135M-Instruct causal language model with
230,400 q/v LoRA parameters. It is not a classical classifier labelled as an
LLM, and a successful result would establish only the measured task and data.

The smaller model makes supervised adaptation practical on a free standard
public CPU runner. Model, tokenizer, chat format, data, code, runtime and
budgets are pinned in `protocol_v10.json`. The original failed 4B model remains
an immutable historical result, not a same-benchmark comparison.

Each listed or referenced order gets a binary next-token score. Identifiers
are masked without interpreting request words or filtering by status. The
decoder selects only one positive candidate; otherwise it escalates. Snapshot
permutation and ID-renaming invariance are structural properties of this
representation, not learned model capabilities. Policy remains a separate veto.

The prefix contains no answer token. Training gathers the last real prefix
hidden state, computes A/B logits, and applies binary cross entropy. All base
weights remain frozen. Training uses 308 unique pairs, including 72 positive
pairs, with balanced sampling. Development uses 24 scenario groups/144 views;
the prospective final uses 36 groups/216 views. Six permutation views per group
are correlated. Their scores reuse identical canonical computations and are
never reported as independent model calls or independent scenarios.

One attempt allows 256 optimizer steps, 2,048 pair presentations and at most
304 scoring computations. The first 16 steps must complete within 600 seconds;
worker wall time and RSS are bounded. Base and checkpoints 128/256 are compared
on development only across the frozen threshold grid. The selected checkpoint
and threshold are frozen before any final inference. Base final scoring uses
its own development-selected threshold. Fresh processes must reproduce saved
development probabilities before final scoring. Both adapters and bounded raw
outputs are retained through logs, without billed Actions artifacts or caches.

References are AI-authored synthetic examples. Exact canonical-prefix overlap
is prohibited across splits; conceptual independence, independent human review,
pretraining contamination and real customer performance are not established.
No independent final qualification is claimed. Final raw intent, actual
controller effects, rules comparisons, latency, cleanup and evidence integrity
must be reported separately. No automatic submission or customer effects occur.

Source: [official model repository](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
