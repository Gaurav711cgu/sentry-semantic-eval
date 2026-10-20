import unittest

class PartitionGuardTests(unittest.TestCase):
    def guard(self):
        try:
            from split_guard import validate_partitions
            return validate_partitions
        except ImportError:
            self.fail('Template-family partition guard missing')

    def case(self,group,family,request):
        return {'group_id':group,'utterance_family_id':family,'request':request,
                'orders':[{'id':'order-a','status':'processing'}],
                'expected_choice':'order-a'}

    def test_disjoint_families_and_prefixes_accepted_without_model_calls(self):
        result=self.guard()({'train':[self.case('t','source-t','Cancel order-a now.')],
                             'development':[self.case('d','source-d','End order-a today.')],
                             'confirmatory':[self.case('f','source-f','Void order-a immediately.')]})
        self.assertEqual(result['groups'],{'train':1,'development':1,'confirmatory':1})

    def test_identifier_renaming_does_not_hide_exact_prefix_leakage(self):
        first=self.case('t','source-t','Cancel order-a now.')
        second=self.case('d','source-d','Cancel order-b now.')
        second['orders']=[{'id':'order-b','status':'processing'}]
        second['expected_choice']='order-b'
        with self.assertRaisesRegex(ValueError,'canonical prefix'):
            self.guard()({'train':[first],'development':[second]})

    def test_paraphrases_from_same_declared_family_cannot_cross_partitions(self):
        with self.assertRaisesRegex(ValueError,'utterance family'):
            self.guard()({'train':[self.case('t','same-source','Cancel order-a now.')],
                          'development':[self.case('d','same-source','Kindly cancel order-a.') ]})

    def test_conflicting_references_inside_permutation_group_rejected(self):
        first=self.case('g','source-g','Cancel order-a now.')
        changed={**first,'expected_choice':None}
        with self.assertRaisesRegex(ValueError,'group reference'):
            self.guard()({'train':[first,changed]})
