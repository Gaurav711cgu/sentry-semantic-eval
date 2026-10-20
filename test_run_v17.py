import unittest
from run_v17 import validate_replay_result

class FreshReferenceResultTests(unittest.TestCase):
    def test_expanded_development_budget_is_bound_to_protocol_not_old_50_prefix_fixture(self):
        p={'optimizer_steps_max':128,'learning_rate':.0001,'dataset':{'sampling_plan_sha256':'a'*64},'runtime':{'torch':'pinned'},'development_scoring_computations':1015}
        r={'passed':True,'completed_optimizer_steps':128,'learning_rate':.0001,'sampling_plan_sha256':'a'*64,'metadata':{'torch':'pinned'},'scoring_requests':1015}
        validate_replay_result(r,p)
        with self.assertRaises(ValueError):validate_replay_result({**r,'scoring_requests':300},p)

    def test_failed_score_response_survives_rejection(self):
        import run_v17
        self.assertTrue(hasattr(run_v17,'retain_score_stage'),'Scoring failure retention missing')
        report={};failure={'passed':False,'failure':'sentinel mismatch'}
        with self.assertRaises(ValueError):run_v17.retain_score_stage(report,'base',failure,{})
        self.assertEqual(report['score_stages']['base'],failure)
