"""Bounded worker planning helpers; no model weights, inference or network."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __init__(self, length=4):
        self.length = length
        self.seen = None

    def encode(self, text, add_special_tokens=False):
        return {"A": [3], "B": [4]}[text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.seen = (messages, tokenize, add_generation_prompt)
        return list(range(1, self.length + 1))


class WorkerTests(unittest.TestCase):
    def test_isolated_writer_directory_allows_parent_cleanup_and_is_one_use(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "adapters"
            self.module().new_adapter_directory(target)
            self.assertEqual(target.stat().st_mode & 0o777, 0o777)
            with self.assertRaises(ValueError):
                self.module().new_adapter_directory(target)

    def module(self):
        try:
            return importlib.import_module("worker_v10")
        except ImportError:
            self.fail("Bounded genuine LoRA worker missing")

    def content(self):
        return json.dumps({"request": "Cancel THIS_ORDER.", "snapshot_count": 3})

    def test_prefix_tokenization_preserves_chat_and_never_adds_training_answer(self):
        tokenizer = FakeTokenizer()
        ids = self.module().encode_prefix(tokenizer, self.content())
        self.assertEqual(ids, [1, 2, 3, 4])
        messages, tokenize, generation = tokenizer.seen
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], self.content())
        self.assertTrue(tokenize)
        self.assertTrue(generation)
        self.assertEqual(self.module().label_ids(tokenizer), [3, 4])

    def test_overlong_prefix_and_nonright_padding_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            self.module().encode_prefix(FakeTokenizer(385), self.content())
        tokenizer = FakeTokenizer()
        tokenizer.padding_side = "left"
        with self.assertRaises(ValueError):
            self.module().encode_prefix(tokenizer, self.content())

    def test_training_labels_are_direct_ce_classes_and_exact_input_schema(self):
        rows = [
            {"prefix_content": self.content(), "label": 0},
            {"prefix_content": self.content(), "label": 1},
        ]
        self.assertEqual(
            self.module().validate_training(
                {"training_pairs": rows, "dev_contents": [self.content()]}
            ),
            (rows, [self.content()]),
        )
        for changed in (
            {"training_pairs": rows[:1], "dev_contents": [self.content()]},
            {
                "training_pairs": [{**rows[0], "label": True}, rows[1]],
                "dev_contents": [self.content()],
            },
            {
                "training_pairs": rows,
                "dev_contents": [self.content()],
                "final_labels": [1],
            },
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.module().validate_training(changed)

    def test_balanced_microbatches_are_deterministic_and_label_preserving(self):
        rows = [{"prefix_content": str(i), "label": i % 2} for i in range(8)]
        first = self.module().balanced_indices(rows, seed=1019, steps=3)
        self.assertEqual(
            first, self.module().balanced_indices(rows, seed=1019, steps=3)
        )
        self.assertEqual(len(first), 3)
        for step in first:
            self.assertEqual(len(step), 4)
            for micro in step:
                self.assertEqual(sorted(rows[i]["label"] for i in micro), [0, 1])

    def test_padding_has_only_prefix_ids_and_masked_rightpad(self):
        ids, mask = self.module().padded([[1, 2, 3], [4]], 0)
        self.assertEqual(ids.tolist(), [[1, 2, 3], [4, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])

    def test_score_adapter_path_stays_inside_owned_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "step128.safetensors").write_bytes(b"fixture")
            self.assertEqual(
                self.module().adapter_path(root, "step128.safetensors"),
                root / "step128.safetensors",
            )
            for name in ("../outside", "/private/tmp/outside", "missing.safetensors"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.module().adapter_path(root, name)

    def test_score_requires_parity_before_any_final_contents(self):
        content = self.content()
        valid = {
            "contents": [content],
            "adapter_file": "step128.safetensors",
            "parity_contents": [content],
            "expected_probabilities": [0.5],
        }
        self.assertEqual(self.module().validate_score(valid), valid)
        for change in (
            {"parity_contents": [], "expected_probabilities": []},
            {"expected_probabilities": [float("nan")]},
            {"final_labels": [0]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module().validate_score({**valid, **change})

    def test_score_label_mapping_is_positive_a_classzero(self):
        self.assertEqual(
            self.module().positive_probability([2.0, -2.0]), 0.9820137900379085
        )

    def test_extreme_logits_do_not_overflow(self):
        self.assertEqual(self.module().positive_probability([-1000.0, 1000.0]), 0.0)
        self.assertEqual(self.module().positive_probability([1000.0, -1000.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
