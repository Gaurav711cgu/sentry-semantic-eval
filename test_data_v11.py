import json
import unittest

from data_v11 import ACTIONS, build_split, training_pairs
from pair_v10 import canonical_pair, pair_inputs


class ContrastiveDataTests(unittest.TestCase):
    def test_every_action_has_positive_and_negated_minimal_pair(self):
        rows = {r["prefix_content"]: r["label"] for r in training_pairs()}
        ids = ["order-red", "order-blue", "order-green"]
        for action in ACTIONS:
            phrase = action.format(t="order-red")
            self.assertEqual(
                rows[canonical_pair("Please " + phrase + ".", ids, "order-red")], 0
            )
            self.assertEqual(
                rows[canonical_pair("Do not " + phrase + ".", ids, "order-red")], 1
            )

    def test_no_class_specific_suffixes_or_conflicting_pair_labels(self):
        rows = training_pairs()
        self.assertGreater(len(rows), 500)
        for suffix in ["", " This concerns a recent purchase."]:
            labels = {
                r["label"]
                for r in rows
                if json.loads(r["prefix_content"])["request"].endswith(suffix)
            }
            self.assertEqual(labels, {0, 1})
        self.assertEqual(len(rows), len({r["prefix_content"] for r in rows}))

    def test_fresh_final_and_development_are_disjoint_from_training(self):
        groups = {}
        contents = {}
        for split in ["dev", "final"]:
            cases = build_split(split)
            groups[split] = {c["group_id"] for c in cases}
            self.assertEqual(len(cases), 144 if split == "dev" else 216)
            contents[split] = {
                r["prefix_content"]
                for c in cases
                for r in pair_inputs(c["request"], [o["id"] for o in c["orders"]])
            }
        train = {r["prefix_content"] for r in training_pairs()}
        self.assertFalse(train & contents["dev"])
        self.assertFalse(train & contents["final"])
        self.assertFalse(contents["dev"] & contents["final"])
        from data_v10 import build_split as old

        old_final = {
            r["prefix_content"]
            for c in old("final")
            for r in pair_inputs(c["request"], [o["id"] for o in c["orders"]])
        }
        self.assertFalse(old_final & contents["final"])


if __name__ == "__main__":
    unittest.main()
