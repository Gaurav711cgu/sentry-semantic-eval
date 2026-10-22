import json
import unittest

from nli_v22 import nli_pair, validate_payload


class ContractTests(unittest.TestCase):
    def test_pair_preserves_scope_and_uses_no_reference_or_policy_filter(self):
        text = 'Do not cancel OTHER_ORDER_0. I endorse "cancel THIS_ORDER" now.'
        premise, hypothesis = nli_pair(json.dumps({'request': text, 'snapshot_count': 3}))
        self.assertIn('Do not cancel Order B.', premise)
        self.assertIn('I endorse "cancel Order A" now.', premise)
        self.assertIn('3', premise)
        self.assertEqual(hypothesis, 'The customer wants to cancel Order A now.')

    def test_invalid_or_duplicate_canonical_fields_fail(self):
        for content in ['{}', '{"request":"x","request":"y","snapshot_count":3}',
                        '{"request":"x","snapshot_count":true}']:
            with self.assertRaises(ValueError):
                nli_pair(content)

    def test_score_payload_cannot_include_labels_or_unbounded_work(self):
        value = {'contents': ['{"request":"Cancel THIS_ORDER","snapshot_count":3}'],
                 'adapter_file': 'step128.safetensors', 'parity_contents': [],
                 'expected_probabilities': []}
        validate_payload('score', value)
        for changed in [dict(value, labels=[0]), dict(value, contents=value['contents'] * 1000),
                        dict(value, adapter_file='../step128.safetensors')]:
            with self.assertRaises(ValueError):
                validate_payload('score', changed)


class TensorTests(unittest.TestCase):
    def test_actual_encoder_gradient_freeze_and_reload(self):
        import copy
        import torch
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
        from nli_v22 import adapt, binary_logits, state, restore
        config = DebertaV2Config(vocab_size=48, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=2,
                                max_position_embeddings=32, relative_attention=False,
                                num_labels=3, pooler_hidden_size=16)
        torch.manual_seed(1019)
        model = DebertaV2ForSequenceClassification(config).float()
        fresh = copy.deepcopy(model)
        adapt(model); adapt(fresh)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        inputs = {'input_ids': torch.tensor([[1, 2, 3], [1, 4, 0]]),
                  'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}
        before = state(model)
        loss = torch.nn.functional.cross_entropy(binary_logits(model(**inputs).logits), torch.tensor([0, 1]))
        loss.backward()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], v) for n, v in state(model).items()))
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]))
        restore(fresh, state(model))
        model.eval(); fresh.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(**inputs).logits, fresh(**inputs).logits, rtol=0, atol=0)
        bad = state(model); bad[next(iter(bad))].fill_(float('nan'))
        with self.assertRaises(ValueError):
            restore(fresh, bad)


if __name__ == '__main__':
    unittest.main()
