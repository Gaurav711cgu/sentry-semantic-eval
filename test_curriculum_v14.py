import unittest

from data_v10 import TRAIN
from data_v13 import build_split as previous_split
from pair_v10 import canonical_pair, pair_inputs


class ReplayCurriculumTests(unittest.TestCase):
    def module(self):
        try:
            import curriculum_v14
            return curriculum_v14
        except ImportError:
            self.fail('Replay curriculum and family-balanced sampling missing')

    def test_original_diverse_training_templates_replayed_without_evaluation_rows(self):
        m = self.module()
        rows, families = m.training_curriculum()
        by_content = {r['prefix_content']: r['label'] for r in rows}
        ids = ['order-red', 'order-blue', 'order-green']
        for kind, templates in TRAIN.items():
            for template in templates:
                request = template.format(t=ids[0], o=ids[1], u='order-missing')
                target = 'order-missing' if kind == 'absent' else ids[0]
                expected = 0 if kind in ['direct', 'distractor', 'absent'] else 1
                key = canonical_pair(request, ids, target)
                self.assertTrue(key in by_content, request)
                self.assertEqual(by_content[key], expected)
        self.assertEqual(len(rows), len(families))
        for split in ['dev', 'final']:
            evaluation = {r['prefix_content'] for c in previous_split(split)
                          for r in pair_inputs(c['request'], [o['id'] for o in c['orders']])}
            self.assertFalse(evaluation & set(by_content))

    def test_balanced_sampler_visits_every_family_and_rotates_within_family(self):
        m = self.module()
        rows = [{'prefix_content': str(i), 'label': i % 2} for i in range(24)]
        families = ['a' if i < 12 else 'b' for i in range(24)]
        plan = m.stratified_indices(rows, families, seed=14, steps=3)
        self.assertEqual(plan, m.stratified_indices(rows, families, seed=14, steps=3))
        seen = {(label, family): [] for label in [0, 1] for family in ['a', 'b']}
        for step in plan:
            self.assertEqual(len(step), 4)
            for batch in step:
                self.assertEqual(sorted(rows[i]['label'] for i in batch), [0, 1])
                for i in batch:
                    seen[rows[i]['label'], families[i]].append(i)
        for values in seen.values():
            self.assertEqual(len(values), 6)
            self.assertEqual(len(set(values)), 6)

    def test_sampling_plan_rejects_wrong_labels_indices_and_family_starvation(self):
        m = self.module()
        rows = [{'prefix_content': str(i), 'label': i % 2} for i in range(8)]
        families = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
        valid = m.stratified_indices(rows, families, seed=1, steps=2)
        m.validate_plan(rows, families, valid, steps=2)
        for bad in [[[[0, 2]] * 4] * 2, [[[0, 99]] * 4] * 2,
                    [[[0, 1]] * 4] * 2]:
            with self.assertRaises(ValueError):
                m.validate_plan(rows, families, bad, steps=2)
