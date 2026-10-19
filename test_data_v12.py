import unittest

from data_v12 import ACTIONS, build_split, training_pairs
from pair_v10 import canonical_pair, pair_inputs


class ScopeSupervisionTests(unittest.TestCase):
    def test_permission_and_scope_minimal_contrasts_cover_every_action(self):
        rows = {r["prefix_content"]: r["label"] for r in training_pairs()}
        ids = ["order-red", "order-blue", "order-green"]
        for action in ACTIONS:
            a = action.format(t="order-red")
            for text, label in [
                ("You have permission to " + a + ".", 0),
                ("You do not have permission to " + a + ".", 1),
                ("Please " + a + ". I do not need this purchase.", 0),
                ("The last action I would want is to " + a + ".", 1),
            ]:
                self.assertEqual(rows[canonical_pair(text, ids, "order-red")], label)

    def test_no_development_or_reserved_final_prefixes_in_training(self):
        train = {r["prefix_content"] for r in training_pairs()}
        for split in ["dev", "final"]:
            contents = {
                r["prefix_content"]
                for c in build_split(split)
                for r in pair_inputs(c["request"], [o["id"] for o in c["orders"]])
            }
            self.assertFalse(train & contents)
        self.assertEqual(len(build_split("final")), 216)


if __name__ == "__main__":
    unittest.main()
