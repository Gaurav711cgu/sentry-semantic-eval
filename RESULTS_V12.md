# V12: eligible recall improves; raw admission still fails

The authorization and negation-scope run completed 128 new optimizer steps and
200 new development scoring computations. The step128 checkpoint at threshold
0.5 reached 126/144 exposed development views (87.5%) and 24/24 eligible-target
hits (100%). There are 24 scenarios with six correlated structural views each.
Raw accuracy remains below 90%; no checkpoint passes both admission gates.
No final inference occurred, no threshold was changed from final results, and
no adapter is promoted.

Compared with v11 on the same development bank, eligible recall increased from
75% to 100%, while overall accuracy stayed at 87.5%. The highest-ranked v12
checkpoint misses both absent-target requests and incorrectly selects an order
in a request conditional on obtaining separate confirmation. This is a change
in the error distribution, not evidence of improved overall generalization.
Further training must address coverage and retention of earlier distinctions,
rather than adding rules to repair a model's predicted targets.

Training took 546.27 seconds with peak RSS 1,686,253,568 bytes. Worker cleanup
and temporary-artifact removal passed. All adapters, scores and the failed
selection are retained in result-v12.json. Independent human/customer validation
is not claimed. See summary-v12.json for exact identities and aggregate values.
