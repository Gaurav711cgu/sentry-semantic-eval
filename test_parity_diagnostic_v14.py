import unittest

class DiagnosticTests(unittest.TestCase):
    def module(self):
        try:
            import parity_diagnostic_v14
            return parity_diagnostic_v14
        except ImportError:
            self.fail('Development-only parity diagnostic missing')

    def test_payload_contains_only_frozen_development_sentinels(self):
        fn=self.module().payload_for
        records=[{'prefix_content':str(i),'positive_probability':.1*i} for i in range(5)]
        value=fn({'adapter_file':None,'records':records})
        self.assertEqual(value['contents'],['0','1','2','3'])
        self.assertEqual(value['parity_contents'],value['contents'])
        self.assertEqual(value['expected_probabilities'],[0,.1,.2,.30000000000000004])
        self.assertEqual(set(value),{'contents','adapter_file','parity_contents','expected_probabilities'})

    def test_failure_output_is_retained_and_does_not_claim_parity(self):
        fn=self.module().stage_record
        failure={'passed':False,'failure':'savedadapter mismatch','worker_exit_code':1}
        result=fn('base',failure)
        self.assertEqual(result['worker'],failure)
        self.assertFalse(result['parity_passed'])
        self.assertFalse(result['final_inference_performed'])
