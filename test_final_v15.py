import unittest
from final_v15 import build_split
from curriculum_v14 import training_pairs
from pair_v10 import pair_inputs

class FrozenFinalTests(unittest.TestCase):
    def test_36_groups_six_permutations_and_no_exact_training_or_development_overlap(self):
        final=build_split('final')
        self.assertEqual(len(final),216)
        groups={c['group_id'] for c in final}
        self.assertEqual(len(groups),36)
        self.assertTrue(all(sum(c['group_id']==g for c in final)==6 for g in groups))
        contents=lambda cases:{r['prefix_content'] for c in cases for r in pair_inputs(c['request'],[o['id'] for o in c['orders']])}
        prefixes=contents(final)
        self.assertFalse(prefixes & contents(build_split('dev')))
        self.assertFalse(prefixes & {r['prefix_content'] for r in training_pairs()})
