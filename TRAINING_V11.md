# Corrective contrastive adaptation

V10 completed execution but failed semantics. Both its retained checkpoints,
consumed budget and base final result remain unchanged. The follow-up fixes a
training-data shortcut: positive/negative scenarios had different suffix banks.

Keep the exact SmolLM2 base, tokenizer, chat format, candidate normalizer,
230,400-parameter q/v adapter and binary prefix-only objective. Initialize from
the v10 step128 adapter, the best **trained** checkpoint by the exposed
development metrics. This choice does not use final results. Reset AdamW state;
use learning rate0.0005, seed1019, microbatch2/accumulation4 and at most128 new
steps. Checkpoints0/32/64/128 get the same exposed development inputs.

Training has880 canonical pairs,154 positive. Every action vocabulary appears
under positive and negative supervision. Minimal contrasts cover negation,
quotation, conditions, withdrawn consent, policy override, multiple targets and
which mentioned order is being preserved. Every class gets the same neutral
augmentation grid. Development stays explicitly exposed. A fresh36-scenario,
216-view AI-authored final is frozen before the corrective run.

No final inference starts unless a development candidate reaches raw accuracy
and eligible-target recall>=90%, with all permutation groups consistent. Select
among admitted candidates by the existing deterministic development ranking.
Freeze threshold/checkpoint before final. Compare the selected checkpoint,
unadapted base and initial v10 checkpoint on identical final inputs, each using
its development-selected threshold. Never tune from that final.

The exact prior base development computations may be reused offline because
model/runtime/prompt/inputs are identical. Bind the complete prior report hash
and disclose50 logical cached computations/zero new physical computations.
Production remains request-local and does not share this evaluation cache.

New budget is separate:128 optimizer steps,1,024 pair presentations,200 fresh
development computations, plus at most three final+parity sets. The protocol
freezes the exact maximum from canonical final input count. No v8/v9/v10
reservation is transferred or reused. Keep zero spending, public standard CPU,
no Actions artifacts/cache, Docker off and heavy inference off the Mac.

Fix evidence retention as well: partial checkpoints and failure details persist
for parent capture; print only a short frozen-selection summary so the normal
log viewer cannot lose subsequent evidence on an oversized line. All v10
evidence was recovered from raw completed-job logs without rerunning inference.

All data remain AI-authored; no independent human/customer validation is claimed.
