import unittest

class ParityEvidenceTests(unittest.TestCase):
    def test_failure_preserves_actual_expected_and_error(self):
        from worker_v21 import parity_evidence
        rows=[{'prefix_content':'example','positive_probability':.4,'duration_seconds':.1}]
        value=parity_evidence(rows,[.5])
        self.assertFalse(value['passed'])
        self.assertEqual(value['records'],rows)
        self.assertEqual(value['expected_probabilities'],[.5])
        self.assertAlmostEqual(value['max_absolute_error'],.1)
        self.assertEqual(value['atol'],1e-6)

    def test_exact_parity_passes_and_missing_evidence_rejected(self):
        from worker_v21 import parity_evidence
        self.assertTrue(parity_evidence([{'positive_probability':.4}],[.4])['passed'])
        for rows,expected in [([],[]),([{'positive_probability':float('nan')}],[.4]),([{'positive_probability':.4}],[])]:
            with self.assertRaises(ValueError):parity_evidence(rows,expected)
