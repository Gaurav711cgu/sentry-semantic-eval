# Replay and family-balanced adaptation

V13 learned one absent-order phrasing but lost distractor recall; no development
checkpoint passed. This experiment changes retention and optimization rather
than adding another verb-template bank. Keep all v13 pairs, replay the original
v10 raw training templates under the same neutral suffixes for both classes,
and add first-letter capitalization variants across every training class.
The old v10 class-specific suffix augmentation is not reused for replay.

Initialize from v12 step128, the highest-ranked v13 initialization. Keep the
same base model, prompt, LoRA architecture, explicit A/B loss and serving code.
Reduce AdamW learning rate from 0.0005 to 0.0001. Training-only family labels
rotate evenly within each class; examples shuffle without replacement within
family cycles. Family metadata never enters model inputs or serving decisions.
Freeze the exact sampling-plan hash before inference and check the retained
worker output against it. Checkpoints are initial/8/16/32/64/128.

One separately consumed run allows 128 new steps, 1,024 presentations and at
most 531 new scoring computations: 300 development plus 231 final/parity.
Fifty prior unadapted-base development computations remain explicit cache.
No final inference unless development raw accuracy and eligible recall reach
90% with all structural groups consistent. Select thresholds/checkpoint on
exposed development only, then compare base/initial/selected once on the same
unused v11 final. All final/controller requirements remain unchanged.

Retain all six adapters and outputs; bound the larger evidence bundle at
16 MiB before replay. No paid runner, Docker, local pretrained inference,
customer effects, rules fallback or independent-validation claim follows.
