"""Real Transformers architecture test with tiny RANDOM tensors, no downloads."""

import copy
import tempfile
import unittest
from pathlib import Path


class ModelStackTests(unittest.TestCase):
    def test_actual_llama_prefix_training_and_safetensors_serving_parity(self):
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import LlamaConfig, LlamaForCausalLM

        from lora_v10 import (
            adapter_state,
            binary_loss,
            install_lora,
            load_adapter,
            pair_logits,
        )

        torch.manual_seed(1019)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            attention_dropout=0.0,
            use_cache=False,
        )
        base = LlamaForCausalLM(config).float()
        fresh = copy.deepcopy(base)
        install_lora(base, rank=2, alpha=4, dropout=0)
        install_lora(fresh, rank=2, alpha=4, dropout=0)
        ids = torch.tensor([[1, 5, 6, 7], [1, 8, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)
        optimizer = torch.optim.AdamW(
            [p for p in base.parameters() if p.requires_grad], lr=0.01
        )
        base.train()
        binary_loss(pair_logits(base, ids, mask, [3, 4]), labels).backward()
        optimizer.step()
        base.eval()
        fresh.eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny-adapter.safetensors"
            save_file(adapter_state(base), str(path))
            load_adapter(fresh, load_file(str(path)))
            with torch.inference_mode():
                torch.testing.assert_close(
                    pair_logits(base, ids, mask, [3, 4]),
                    pair_logits(fresh, ids, mask, [3, 4]),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
