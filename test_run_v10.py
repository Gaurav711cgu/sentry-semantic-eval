import hashlib
import tempfile
import unittest
from pathlib import Path

from run_v10 import checked_json, retained_file, validate_training_grant, verify_sources


class RunContracts(unittest.TestCase):
    def test_reject_duplicate_fields_and_nonfinite(self):
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}']:
            with self.assertRaises(ValueError):
                checked_json(raw)

    def test_source_hashes_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.py").write_bytes(b"original")
            pins = {"source.py": hashlib.sha256(b"original").hexdigest()}
            verify_sources(root, pins)
            (root / "source.py").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_sources(root, pins)
            for unsafe in ["../source.py", "/tmp/source.py"]:
                with self.assertRaises(ValueError):
                    verify_sources(root, {unsafe: "0" * 64})

    def test_consumed_grant_exact_run_source_protocol_and_budget(self):
        expected = {
            "experiment_id": "v10",
            "source_head": "a" * 40,
            "protocol_sha256": "b" * 64,
            "optimizer_steps_max": 256,
        }
        grant = dict(
            expected, run_id=123, attempt=1, status="consumed_before_execution"
        )
        validate_training_grant(grant, expected, 123, 1)
        for field, value in [
            ("run_id", 124),
            ("attempt", 2),
            ("optimizer_steps_max", 257),
            ("source_head", "c" * 40),
            ("status", "reserved"),
        ]:
            changed = dict(grant, **{field: value})
            with self.assertRaises(ValueError):
                validate_training_grant(changed, expected, 123, 1)

    def test_retention_hash_and_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.safetensors"
            path.write_bytes(b"adapter-tensors")
            record = retained_file(path, 100)
            self.assertEqual(
                record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(record["bytes"], 15)
            with self.assertRaises(ValueError):
                retained_file(path, 2)


if __name__ == "__main__":
    unittest.main()
