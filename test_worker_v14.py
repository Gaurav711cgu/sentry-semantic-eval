import unittest


class ReplayInputTests(unittest.TestCase):
    def module(self):
        try:
            import worker_v14
            return worker_v14
        except ImportError:
            self.fail('Validated replay worker input missing')

    def value(self):
        import json
        return {
            'training_pairs': [
                {'prefix_content': json.dumps({'request': f'Example {i} THIS_ORDER', 'snapshot_count': 3}), 'label': i % 2}
                for i in range(8)],
            'training_families': ['direct', 'direct', 'absent', 'absent', 'direct', 'direct', 'absent', 'absent'],
            'dev_contents': [json.dumps({'request': 'Inspect THIS_ORDER', 'snapshot_count': 3})],
            'initial_adapter_file': 'initial.safetensors',
            'initial_adapter_sha256': 'a' * 64,
        }

    def test_training_only_families_produce_bounded_balanced_plan_before_loading_weights(self):
        rows, dev, plan = self.module().validate_replay_input(self.value())
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(dev), 1)
        self.assertEqual(len(plan), 128)
        self.assertEqual(sum(len(batch) for step in plan for batch in step), 1024)
        for step in plan:
            for batch in step:
                self.assertEqual(sorted(rows[i]['label'] for i in batch), [0, 1])

    def test_missing_misaligned_or_nontext_family_rejected_before_loading_weights(self):
        module = self.module()
        for families in [[], ['direct'] * 7, ['direct'] * 7 + [False]]:
            with self.assertRaises(ValueError):
                module.validate_replay_input({**self.value(), 'training_families': families})
        with self.assertRaises(ValueError):
            module.validate_replay_input({**self.value(), 'final_labels': [1]})
