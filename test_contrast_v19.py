import unittest

class ContrastTests(unittest.TestCase):
    def test_paired_training_and_heldout_constructions_are_separate(self):
        from contrast_v19 import curriculum, development, paired_indices
        from data_v17 import training_curriculum
        from pair_v10 import pair_inputs
        rows, families, pairs = curriculum()
        old, _ = training_curriculum()
        self.assertEqual(rows[:len(old)], old)
        self.assertGreater(len(pairs), 200)
        self.assertTrue(all(rows[a]['label'] == 0 and rows[b]['label'] == 1 for a,b in pairs))
        dev = development()
        self.assertTrue(all(c['id'].startswith('dev-') for c in dev))
        contents = {r['prefix_content'] for c in dev for r in pair_inputs(c['request'],[o['id'] for o in c['orders']])}
        self.assertFalse({r['prefix_content'] for r in rows} & contents)
        plan = paired_indices(rows, families)
        self.assertEqual(len(plan),128)
        self.assertEqual(sum(len(b) for s in plan for b in s),1024)
        self.assertTrue(all(sorted(rows[i]['label'] for i in b)==[0,1] for s in plan for b in s))
        self.assertEqual(plan,paired_indices(rows,families))

    def test_plan_rejects_modified_corpus(self):
        from contrast_v19 import curriculum,paired_indices
        rows,families,_=curriculum()
        with self.assertRaises(ValueError):paired_indices(rows[:-1],families[:-1])

    def test_training_does_not_copy_reserved_or_consumed_finals(self):
        from contrast_v19 import curriculum
        from data_v17 import build_split
        from final_v15 import build_split as old_final
        from pair_v10 import pair_inputs
        training={r['prefix_content'] for r in curriculum()[0]}
        for cases in [build_split('final'),old_final('final')]:
            final={r['prefix_content'] for c in cases for r in pair_inputs(c['request'],[o['id'] for o in c['orders']])}
            self.assertFalse(training & final)
