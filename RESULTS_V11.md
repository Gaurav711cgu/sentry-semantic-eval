# V11: development improves, admission still fails

The corrective run completed all 128 new optimizer steps and 200 new development
scoring computations. Its best checkpoint reached 126/144 exposed development
views (87.5%), including 18/24 eligible targets (75%). Both are below the frozen
90% requirements. No final inference occurred, and no model was promoted.

Compared with the v10 trained checkpoints on this same development bank, raw
accuracy improved from 66.7% to 87.5%. This is a development improvement, not
an independent final result. The three failed scenario groups concern explicit
permission, unrelated negation in an absent-target request, and an indirect
negation idiom. All checkpoints and probabilities remain retained.

The next curriculum addresses these semantic families using minimal contrasts
across every action verb, including active/passive wording. It does not copy
development rows, change the decoder/controller, or touch final labels. The
reserved v11 final remains unused and frozen. See `summary-v11.json` and
`result-v11.json`.
