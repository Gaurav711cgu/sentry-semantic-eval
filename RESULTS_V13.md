# V13: completed training, no development improvement

All 128 new optimizer steps and 200 new development computations completed.
The highest-ranked checkpoint is the unchanged v12 initialization: 126/144 raw
exact (87.5%), 24/24 eligible-target hits (100%). No newly trained checkpoint
passes admission; no final inference or model promotion occurred.

New supervision taught the second absent-order phrasing, but the highest-ranked
new checkpoint loses both distractor requests and still misses the first absent
request. Its best raw result remains 126/144 with 12/24 eligible hits. Training
loss is low on several logged minibatches, which does not establish corpus-wide
fit or generalization. Among 2,758 available pairs, class-balanced sampling with
replacement presents 801 distinct pairs in 1,024 presentations (355 positive,
446 negative). Later curricula preserve v11/v12 supervision but do not replay
the earlier, more varied v10 corpus. These are concrete coverage and retention
issues to investigate; adding templates did not demonstrate an overall repair.

Training took 825.27 seconds with peak RSS 1,637,519,360 bytes. Owned-worker stop
and temporary-artifact removal passed. No memory crash, timeout or quota
interruption explains this run's semantic result. A faster paid runner alone
would not change the same completed optimization. Model-capacity sufficiency
has not been established. All checkpoints and scores remain retained.

The prospective final remains untouched. References remain AI-authored synthetic;
no independent human/customer validation is claimed. See summary-v13.json and
result-v13.json for exact results and retained identities.
