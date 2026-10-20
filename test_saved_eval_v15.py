import unittest

class RetainedStageTests(unittest.TestCase):
    def helper(self):
        try:
            from saved_eval_v15 import retain_stage
            return retain_stage
        except ImportError:
            self.fail('Persisted score-stage results missing')

    def test_failure_retained_before_rejection_and_final_not_marked_complete(self):
        value={'stages':{}}
        failure={'passed':False,'failure':'parity mismatch'}
        with self.assertRaises(ValueError):self.helper()(value,'base',failure)
        self.assertEqual(value['stages']['base'],failure)

    def test_success_requires_exact_runtime_and_parity(self):
        fn=self.helper(); value={'stages':{}}
        good={'passed':True,'parity':{'passed':True},'metadata':{'python':'pinned'}}
        fn(value,'selected',good,{'python':'pinned'})
        with self.assertRaises(ValueError):fn(value,'bad',{**good,'metadata':{}},{'python':'pinned'})
