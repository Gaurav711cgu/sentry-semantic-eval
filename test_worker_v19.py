import copy
import unittest
from worker_v19 import validate_replay_input
from contrast_v19 import curriculum,paired_indices,development
from run_v19 import unique_contents

class WorkerTests(unittest.TestCase):
    def payload(self):
        rows,families,_=curriculum()
        return {'training_pairs':rows,'training_families':families,'training_plan':paired_indices(rows,families),
                'dev_contents':unique_contents(development()),'initial_adapter_file':'initial.safetensors','initial_adapter_sha256':'a'*64}

    def test_exact_plan_and_all_class_pairs(self):
        value=self.payload()
        rows,dev,plan=validate_replay_input(value)
        self.assertEqual(len(plan),128)
        self.assertEqual(len(rows),len(value['training_pairs']))
        self.assertGreater(len(dev),145)

    def test_corrupted_plans_rejected(self):
        for bad in [True,-1,999999]:
            value=copy.deepcopy(self.payload());value['training_plan'][0][0][0]=bad
            with self.assertRaises(ValueError):validate_replay_input(value)
        value=self.payload();value['training_plan']=value['training_plan'][:127]
        with self.assertRaises(ValueError):validate_replay_input(value)
