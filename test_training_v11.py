import hashlib
import tempfile
import unittest
from pathlib import Path

from run_v11 import admission_choice
from worker_v11 import verified_initial


class CorrectiveTrainingTests(unittest.TestCase):
    def test_initial_adapter_identity_is_exact_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "initial.safetensors"
            p.write_bytes(b"owned-adapter")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(verified_initial(root, p.name, digest), p)
            for name, sha in [("../outside", digest), (p.name, "a" * 64)]:
                with self.assertRaises(ValueError):
                    verified_initial(root, name, sha)

    def test_no_final_scoring_candidate_without_development_admission(self):
        failed = {
            "name": "base",
            "step": 0,
            "threshold": 0.6,
            "metrics": {
                "raw_accuracy": 0.75,
                "eligible_recall": 0.0,
                "consistent_groups": 24,
                "permutation_groups": 24,
            },
        }
        self.assertIsNone(admission_choice({"candidates": [failed]}))
        passing = {
            "name": "step64",
            "step": 64,
            "threshold": 0.8,
            "metrics": {
                "raw_accuracy": 0.95,
                "eligible_recall": 1.0,
                "consistent_groups": 24,
                "permutation_groups": 24,
            },
        }
        self.assertEqual(
            admission_choice({"candidates": [failed, passing]})["name"], "step64"
        )


if __name__ == "__main__":
    unittest.main()
