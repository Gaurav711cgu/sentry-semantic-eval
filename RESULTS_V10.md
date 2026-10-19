# Supervised v10: reject semantic promotion

[Run 39294671400](https://github.com/Gaurav711cgu/sentry-semantic-eval/actions/runs/39294671400)
completed 256 CPU optimizer steps and retained both genuine adapters. Training
took 1,460.32 seconds including checkpoint development scoring; first 16 steps
took 91.00 seconds. Peak RSS was 1,641,484,288 bytes. Cleanup and fresh-process
base scoring passed. A green workflow here means execution, not semantic quality.

The trained checkpoints reached 96/144 development raw decisions (66.7%) and
12/24 eligible targets. The fixed ranking correctly preferred the unadapted
base control, whose final score was 162/216 (75%), with no eligible-target hits.
No trained adapter was promoted or scored on this final. Near-zero training
loss did not establish generalization.

The training generator used different suffix families for positive and negative
scenario types, and its development bank changed lexical patterns substantially.
This permits style shortcuts instead of learning the intended contrast. Both
failed adapters and every score remain retained. The follow-up corrects the
training corpus with shared neutral augmentation and positive/negative minimal
pairs; it keeps the base model, normalizer, objective and adapter architecture.

Budget: 256/256 optimizer steps and 227 new scoring computations completed, from
a maximum reservation of 304. No remaining reservation is transferred to a
new run. Final inputs were AI-authored synthetic scenarios, not independent
human or customer validation. See `summary-v10.json` and `result-v10.json`.
