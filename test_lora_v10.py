"""Genuine LoRA arithmetic on tiny tensors; no model acquisition."""

import importlib
import unittest

import torch
from torch import nn


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.embedding = nn.Embedding(10, 4)
        self.seen = None

    def forward(self, input_ids, attention_mask):
        self.seen = (input_ids.clone(), attention_mask.clone())
        hidden = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        hidden = hidden + torch.arange(input_ids.shape[1]).view(1, -1, 1)
        return type("Output", (), {"last_hidden_state": hidden})()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(4, 7, bias=False)


class LoraTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("lora_v10")
        except ImportError:
            self.fail("Genuine q/v projection LoRA implementation missing")

    def test_adapter_freezes_base_preserves_initial_output_and_receives_gradients(self):
        m = self.module()
        model = TinyModel()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        before = model.model.q_proj(x).detach().clone()
        base = list(model.parameters())
        self.assertIs(m.install_lora(model), model)
        self.assertTrue(all(not p.requires_grad for p in base))
        self.assertEqual(
            {n for n, p in model.named_parameters() if p.requires_grad},
            {
                "model.q_proj.lora_A",
                "model.q_proj.lora_B",
                "model.v_proj.lora_A",
                "model.v_proj.lora_B",
            },
        )
        torch.testing.assert_close(before, model.model.q_proj(x))
        self.assertEqual(model.model.q_proj.scaling, 2.0)
        self.assertEqual(model.model.q_proj.dropout.p, 0.05)
        model.model.q_proj(x).sum().backward()
        self.assertGreater(model.model.q_proj.lora_B.grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.model.q_proj.base.weight.grad)

    def test_smol_projection_dimensions_yield_exact230400_trainable_parameters(self):
        model = nn.Module()
        model.layers = nn.ModuleList()
        for _ in range(30):
            layer = nn.Module()
            layer.q_proj = nn.Linear(576, 576, bias=False)
            layer.v_proj = nn.Linear(576, 192, bias=False)
            model.layers.append(layer)
        self.module().install_lora(model)
        self.assertEqual(
            sum(p.numel() for p in model.parameters() if p.requires_grad), 230400
        )

    def test_addition_is_actual_low_rank_matrix_product(self):
        model = TinyModel()
        self.module().install_lora(model, rank=2, alpha=4, dropout=0.0)
        layer = model.model.q_proj
        with torch.no_grad():
            layer.lora_A.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
            layer.lora_B.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            )
        x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        torch.testing.assert_close(
            layer(x), layer.base(x) + 2.0 * (x @ layer.lora_A.T @ layer.lora_B.T)
        )

    def test_prefix_only_last_real_position_logits_no_answerappend_or_doubleshift(self):
        model = TinyModel()
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        expected_hidden = torch.tensor([[1.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        expected = model.lm_head(expected_hidden)[:, [1, 5]]
        actual = self.module().pair_logits(model, ids, mask, [1, 5])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.model.seen[0], ids)
        torch.testing.assert_close(model.model.seen[1], mask)

    def test_reject_leftpadding_holes_empty_rows_and_invalid_label_tokens(self):
        model = TinyModel()
        ids = torch.tensor([[1, 2, 0]])
        for mask in (
            torch.tensor([[0, 1, 1]]),
            torch.tensor([[1, 0, 1]]),
            torch.tensor([[0, 0, 0]]),
        ):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.module().pair_logits(model, ids, mask, [1, 5])
        for labels in ([1, 1], [-1, 5], [1, 7], [1], [True, 5]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.module().pair_logits(model, ids, torch.ones_like(ids), labels)

    def test_binary_loss_is_direct_class_ce_and_validates_labels(self):
        logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        result = self.module().binary_loss(logits, labels)
        torch.testing.assert_close(
            result, torch.nn.functional.cross_entropy(logits, labels)
        )
        result.backward()
        self.assertEqual(logits.grad.shape, (2, 2))
        for bad in (
            torch.tensor([0, 2]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([[0, 1]]),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.module().binary_loss(logits, bad)

    def test_only_adapter_tensors_export_and_fresh_model_reload_exact_parity(self):
        m = self.module()
        torch.manual_seed(123)
        original = TinyModel()
        base = {k: v.clone() for k, v in original.state_dict().items()}
        m.install_lora(original, dropout=0.0)
        with torch.no_grad():
            for _, p in original.named_parameters():
                if p.requires_grad:
                    p.fill_(0.25)
        state = m.adapter_state(original)
        self.assertTrue(state)
        self.assertTrue(all(k.endswith((".lora_A", ".lora_B")) for k in state))
        fresh = TinyModel()
        fresh.load_state_dict(base)
        m.install_lora(fresh, dropout=0.0)
        m.load_adapter(fresh, state)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(
            original.model.q_proj(x), fresh.model.q_proj(x), rtol=0, atol=0
        )
        state[next(iter(state))].zero_()
        self.assertTrue(
            any(
                p.abs().sum() > 0
                for n, p in fresh.named_parameters()
                if n.endswith("lora_A")
            )
        )

    def test_load_reject_unknown_missing_shape_nonfinite_before_any_copy(self):
        m = self.module()
        model = TinyModel()
        m.install_lora(model)
        good = m.adapter_state(model)
        first = next(iter(good))
        variants = [
            {},
            {**good, "base.weight": torch.zeros(1)},
            {**good, first: torch.zeros(1)},
            {**good, first: torch.full_like(good[first], float("nan"))},
        ]
        for state in variants:
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                m.load_adapter(model, state)
            for key, value in m.adapter_state(model).items():
                torch.testing.assert_close(value, good[key])

    def test_invalid_install_and_duplicate_install_rejected(self):
        m = self.module()
        for kwargs in ({"rank": 0}, {"rank": True}, {"alpha": 0}, {"dropout": 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                m.install_lora(TinyModel(), **kwargs)
        model = TinyModel()
        m.install_lora(model)
        with self.assertRaises(ValueError):
            m.install_lora(model)

    def test_tiny_nexttoken_training_updates_only_lora_and_reload_pair_logits_match(
        self,
    ):
        class TrainBackbone(TinyBackbone):
            def forward(self, input_ids, attention_mask):
                embedded = self.embedding(input_ids)
                hidden = self.q_proj(embedded) + torch.nn.functional.pad(
                    self.v_proj(embedded), (0, 2)
                )
                return type("Output", (), {"last_hidden_state": hidden})()

        def fresh():
            value = TinyModel()
            value.model = TrainBackbone()
            return value

        m = self.module()
        torch.manual_seed(321)
        model = fresh()
        base = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        m.install_lora(model, dropout=0.0)
        ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        before = m.pair_logits(model, ids, mask, [1, 5]).detach().clone()
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        optimizer.zero_grad()
        m.binary_loss(
            m.pair_logits(model, ids, mask, [1, 5]), torch.tensor([0, 1])
        ).backward()
        optimizer.step()
        after = m.pair_logits(model, ids, mask, [1, 5]).detach()
        self.assertFalse(torch.equal(before, after))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                original_name = name.replace(".q_proj.base.", ".q_proj.").replace(
                    ".v_proj.base.", ".v_proj."
                )
                torch.testing.assert_close(
                    parameter, base[original_name], atol=0, rtol=0
                )
        reloaded = fresh()
        reloaded.load_state_dict(base)
        m.install_lora(reloaded, dropout=0.0)
        m.load_adapter(reloaded, m.adapter_state(model))
        torch.testing.assert_close(
            after, m.pair_logits(reloaded, ids, mask, [1, 5]), rtol=0, atol=0
        )

    def test_install_preserves_eval_mode_and_converts_base_to_cpu_fp32(self):
        model = TinyModel().double().eval()
        self.module().install_lora(model)
        self.assertFalse(model.model.q_proj.training)
        self.assertTrue(
            all(
                p.dtype == torch.float32 and p.device.type == "cpu"
                for p in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
