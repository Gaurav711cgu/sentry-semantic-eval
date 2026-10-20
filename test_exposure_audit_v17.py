import unittest
from exposure_audit_v17 import exposure, error_kind

class AuditTests(unittest.TestCase):
    def test_exposure_counts_repeated_presentations_separately(self):
        rows = [{'label': 0}, {'label': 1}]
        value = exposure(rows, ['direct', 'quoted'], [[[0, 1], [0, 1]]], 1)
        self.assertEqual(value['presentations'], 4)
        self.assertEqual(value['unique_pairs'], 2)
        self.assertEqual(value['class_family'][0]['unique_pairs'], 1)
        with self.assertRaises(ValueError):
            exposure(rows, ['direct', 'quoted'], [], 1)

    def test_distinguishes_error_directions(self):
        self.assertEqual(error_kind(None, 'a', 1), 'false_authorization')
        self.assertEqual(error_kind('a', None, 0), 'missed_authorization')
        self.assertEqual(error_kind('a', None, 2), 'multiple_positive_abstention')
        self.assertEqual(error_kind('a', 'b', 1), 'wrong_target')
        self.assertEqual(error_kind(None, None, 0), 'correct')
