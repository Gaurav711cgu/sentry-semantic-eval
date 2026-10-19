import unittest


class ReplayResultGateTests(unittest.TestCase):
    def validator(self):
        try:
            from run_v14 import validate_replay_result
            return validate_replay_result
        except ImportError:
            self.fail('Replay result gate missing')

    def values(self):
        protocol = {'optimizer_steps_max': 128, 'learning_rate': 0.0001,
                    'runtime': {'torch': 'pinned'},
                    'dataset': {'sampling_plan_sha256': 'a' * 64}}
        result = {'passed': True, 'completed_optimizer_steps': 128,
                  'learning_rate': .0001, 'metadata': {'torch': 'pinned'},
                  'sampling_plan_sha256': 'a' * 64, 'scoring_requests': 300}
        return result, protocol

    def test_exact_complete_replay_result_admitted(self):
        result, protocol = self.values()
        self.validator()(result, protocol)

    def test_drift_in_optimizer_plan_runtime_or_budget_rejected(self):
        fn = self.validator()
        result, protocol = self.values()
        for key, value in [('learning_rate', .0005), ('sampling_plan_sha256', 'b' * 64),
                           ('scoring_requests', 301), ('completed_optimizer_steps', 127),
                           ('metadata', {'torch': 'different'}), ('passed', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                fn({**result, key: value}, protocol)
