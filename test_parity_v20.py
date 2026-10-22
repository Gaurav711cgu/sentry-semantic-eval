import base64
import hashlib
import json
from pathlib import Path
import unittest
from parity_diagnostic_v20 import payload_for
from worker_v10 import validate_score

class ParityTests(unittest.TestCase):
    def test_exact_selected_adapter_and_bounded_retained_sentinels(self):
        packet=json.loads(Path(__file__).with_name('parity-input-v20.json').read_text())
        checkpoint=packet['checkpoint'];record=packet['retained_adapter']
        raw=base64.b64decode(record['base64'],validate=True)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),record['sha256'])
        self.assertEqual(record['sha256'],checkpoint['adapter_sha256'])
        self.assertEqual(len(raw),record['bytes'])
        self.assertEqual(checkpoint['step'],64)
        value=payload_for(checkpoint);validate_score(value)
        self.assertEqual(len(value['contents']),4)
        self.assertEqual(len(value['parity_contents']),4)
        self.assertEqual(value['contents'],value['parity_contents'])
        self.assertNotIn('label',value)
