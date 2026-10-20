import collections
import unittest

class GeneralizationDataTests(unittest.TestCase):
    def module(self):
        try:
            import data_v17
            return data_v17
        except ImportError:
            self.fail('Broader family-separated language corpus missing')

    def test_development_has_six_distinct_groups_in_each_challenge_category(self):
        m=self.module();cases=m.build_split('dev')
        groups={c['group_id']:c for c in cases}
        self.assertEqual(len(groups),72)
        self.assertEqual(set(collections.Counter(c['kind'] for c in groups.values()).values()),{6})
        self.assertEqual(len(cases),432)

    def test_all_partitions_pass_family_and_canonical_leakage_guard(self):
        from split_guard import validate_partitions
        m=self.module()
        checked=validate_partitions({s:m.annotated_split(s) for s in ['train','dev','final']})
        self.assertEqual(checked['groups']['dev'],72)
        self.assertEqual(checked['groups']['final'],36)

    def test_old_training_pairs_preserved_without_contradictory_labels(self):
        from curriculum_v14 import training_pairs
        m=self.module();rows={r['prefix_content']:r['label'] for r in m.training_curriculum()[0]}
        for row in training_pairs():self.assertEqual(rows[row['prefix_content']],row['label'])
        self.assertGreater(len(rows),3076)

    def test_consumed_v15_final_never_copied_into_new_training_or_development(self):
        from final_v15 import build_split
        from pair_v10 import pair_inputs
        m=self.module()
        old={r['prefix_content'] for c in build_split('final') for r in pair_inputs(c['request'],[o['id'] for o in c['orders']])}
        for split in ['train','dev']:
            prefixes={r['prefix_content'] for c in m.build_split(split) for r in pair_inputs(c['request'],[o['id'] for o in c['orders']])}
            self.assertFalse(old & prefixes)
