import unittest

class LearningDiagnosisTests(unittest.TestCase):
    def module(self):
        try:
            import learning_diagnosis
            return learning_diagnosis
        except ImportError:
            self.fail('Learning coverage diagnosis missing')

    def test_coverage_counts_presentations_unique_pairs_and_unseen_families(self):
        rows=[{'label':0},{'label':1},{'label':0},{'label':1}]
        result=self.module().coverage(rows,['direct','negated','absent','quoted'],[[[0,1],[0,1]]],1)
        self.assertEqual(result['presentations'],4)
        self.assertEqual(result['unique_pairs'],2)
        self.assertEqual(result['unseen_pairs'],2)
        self.assertEqual(result['families']['absent']['presentations'],0)
        self.assertEqual(result['families']['direct']['presentations'],2)

    def test_out_of_range_indices_and_invalid_checkpoint_rejected(self):
        fn=self.module().coverage
        for step,plan in [(0,[[[0,1]]]),(2,[[[0,1]]]),(1,[[[0,9]]])]:
            with self.assertRaises(ValueError):fn([{'label':0},{'label':1}],['a','b'],plan,step)

    def test_result_summary_separates_scenario_count_from_correlated_views(self):
        m={'cases':12,'permutation_groups':2,'raw_exact_correct':6,'eligible_raw_target_hits':6,'eligible_targets':6,'group_results':[{'group_id':'a','exact':False},{'group_id':'b','exact':True}]}
        result=self.module().metric_summary(m)
        self.assertEqual(result['scenario_groups'],2)
        self.assertEqual(result['correlated_views'],12)
        self.assertEqual(result['wrong_groups'],['a'])

    def test_probe_is_deterministic_bounded_and_preserves_label_membership(self):
        fn=self.module().probe_sample
        rows=[{'prefix_content':str(i),'label':i%2} for i in range(100)]
        families=['direct' if i<50 else 'negated' for i in range(100)]
        plan=[[[i,i+1]] for i in range(0,16,2)]
        first=fn(rows,families,plan)
        self.assertEqual(first,fn(rows,families,plan))
        self.assertLessEqual(len(first),32)
        for row in first:
            self.assertEqual(row['label'],rows[row['training_index']]['label'])
            self.assertEqual(row['presented_before_step8'],row['training_index']<16)
