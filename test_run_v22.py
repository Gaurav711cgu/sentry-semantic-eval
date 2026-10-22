import unittest
from contrast_v19 import curriculum, development, paired_indices
from confirmation_v22 import cases
from evaluate_v10 import score_cases
from nli_v22 import validate_payload
from run_v19 import unique_contents
from run_v22 import stage_ok


class CampaignTests(unittest.TestCase):
    def test_unchanged_corpus_and_disjoint_confirmation(self):
        rows, families, _ = curriculum()
        dev = unique_contents(development()); final = unique_contents(cases())
        self.assertEqual((len(rows), len(dev), len(cases()), len(final)), (6787, 195, 144, 49))
        self.assertFalse(set(final) & (set(dev) | {r['prefix_content'] for r in rows}))
        validate_payload('train', {'training_pairs': rows, 'training_families': families,
                                  'training_plan': paired_indices(rows, families), 'dev_contents': dev})
        metrics = score_cases(cases(), {content: .0 for content in final}, .8)
        self.assertEqual(metrics['permutation_groups'], 24)
        self.assertEqual(metrics['eligible_targets'], 60)
        self.assertLess(metrics['raw_accuracy'], .9)

    def test_failure_runtime_and_memory_reject_before_selection(self):
        p = {'runtime': {'torch': '2.8.0+cpu'}}
        good = {'passed': True, 'rss_peak_bytes': 1, 'metadata': p['runtime']}
        self.assertTrue(stage_ok(good, p))
        for changed in [dict(good, passed=False), dict(good, metadata={}),
                        dict(good, rss_peak_bytes=9 * 1024**3), {}]:
            self.assertFalse(stage_ok(changed, p))


if __name__ == '__main__':
    unittest.main()
