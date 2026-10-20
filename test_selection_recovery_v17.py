import unittest

class IdentityRecoveryTests(unittest.TestCase):
    def converter(self):
        try:
            from selection_recovery_v17 import development_cases
            return development_cases
        except ImportError:
            self.fail('Lossless development identity recovery missing')

    def test_identity_conversion_preserves_every_model_input_and_reference(self):
        from data_v17 import build_split
        from pair_v10 import pair_inputs
        old=build_split('dev');new=self.converter()(old)
        self.assertEqual(len(old),len(new))
        for a,b in zip(old,new):
            self.assertTrue(b['id'].startswith('dev-'))
            self.assertEqual({k:v for k,v in a.items() if k not in ['id','group_id']},{k:v for k,v in b.items() if k not in ['id','group_id']})
            self.assertEqual(pair_inputs(a['request'],[o['id'] for o in a['orders']]),pair_inputs(b['request'],[o['id'] for o in b['orders']]))

    def test_final_cases_cannot_be_reclassified_as_development(self):
        from data_v17 import build_split
        with self.assertRaises(ValueError):self.converter()(build_split('final'))
