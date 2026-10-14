"""Tiny-model gradient/readout check; optional training dependencies, no downloads required."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Training environment is optional")
class NativeTrainingContract(unittest.TestCase):
    def test_slot_loss_updates_adapters_and_reloads(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from peft import LoraConfig, get_peft_model
        from train_classifier import slot_logits

        torch.manual_seed(13)
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=32, linear_num_key_heads=1, linear_num_value_heads=2,
                                 linear_key_head_dim=32, linear_value_head_dim=32,
                                 layer_types=["linear_attention", "full_attention"],
                                 pad_token_id=0, use_cache=False)
        base = Qwen3_5ForCausalLM(config)
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                               target_modules="all-linear", lora_dropout=0))
        ids = torch.randint(1, 128, (1, 16))
        letters = torch.tensor([2, 3])
        model.eval()
        with torch.no_grad():
            restricted = slot_logits(model, ids, 15, letters)
            complete = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, 15, letters]
        torch.testing.assert_close(restricted, complete)
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        model.train()
        optimizer = torch.optim.AdamW(trainable, lr=.001)
        loss = torch.nn.functional.cross_entropy(slot_logits(model, ids, 15, letters), torch.tensor([1]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable))
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, trainable)))
        model.eval()
        with torch.no_grad():
            expected = slot_logits(model, ids, 15, letters)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, safe_serialization=True)
            model.load_adapter(folder, adapter_name="reloaded", is_trainable=False)
            model.set_adapter("reloaded")
            model.eval()
            with torch.no_grad():
                actual = slot_logits(model, ids, 15, letters)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
