import unittest
from study_v18 import build_study, select_probes

class StudyTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_does_not_include_final(self):
        first = build_study()
        self.assertEqual(first, build_study())
        self.assertFalse(first['final_inputs_included'])
        self.assertEqual(first['next_execution']['training_steps'], 0)
        probes = first['fit_probes']
        self.assertEqual(len(probes), len({p['prefix_content'] for p in probes}))
        self.assertTrue({p['presented_in_v17_first32'] for p in probes} == {True, False})
        self.assertEqual(first['next_execution']['maximum_physical_scoring_requests'], 3 * (len(probes) + 4))

    def test_invalid_probe_budget_rejected(self):
        for value in [0, 9, True]:
            with self.assertRaises(ValueError):
                select_probes([], [], [], value)

    def test_retained_adapters_and_parity_inputs_are_available(self):
        import json, base64, hashlib
        from pathlib import Path
        from fit_diagnostic_v18 import payload_for
        from worker_v10 import validate_score
        report = json.loads(Path(__file__).with_name('result-v17.json').read_text())
        probes = build_study()['fit_probes']
        for name in ['initial', 'step32', 'step128']:
            item = report['retained_adapters'][name]
            raw = base64.b64decode(item['base64'], validate=True)
            self.assertEqual(len(raw), item['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
            value = payload_for(report['training']['checkpoints'][name])
            value['contents'] = [p['prefix_content'] for p in probes]
            validate_score(value)
            self.assertEqual(len(value['parity_contents']), 4)
