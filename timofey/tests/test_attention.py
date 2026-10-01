"""Equivalence and masking checks for explicit-KV SDPA; no pretrained weights."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Phi3Config, Phi3ForCausalLM
from transformers.masking_utils import AttentionMaskInterface

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phi.attention import ATTENTION_IMPLEMENTATION, register_attention, sdpa_repeat_kv


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = SimpleNamespace(is_causal=True)
        self.query = torch.randn(2, 4, 6, 8, requires_grad=True)
        self.key = torch.randn(2, 2, 6, 8, requires_grad=True)
        self.value = torch.randn(2, 2, 6, 8, requires_grad=True)

    def reference(self, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            self.query, self.key, self.value, attn_mask=mask,
            is_causal=mask is None, enable_gqa=True).transpose(1, 2).contiguous()

    def test_causal_gqa_output_and_gradient_equivalence(self):
        actual, weights = sdpa_repeat_kv(self.module, self.query, self.key, self.value, None)
        expected = self.reference(None)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertIsNone(weights)
        actual_grads = torch.autograd.grad(actual.square().sum(), (self.query, self.key, self.value), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (self.query, self.key, self.value))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    def test_padding_mask_equivalence(self):
        mask = torch.ones(2, 1, 6, 6, dtype=torch.bool).tril()
        mask[0, :, :, :2] = False
        actual, _ = sdpa_repeat_kv(self.module, self.query, self.key, self.value, mask)
        torch.testing.assert_close(actual, self.reference(mask), rtol=1e-5, atol=1e-6)

    def test_never_requests_sdpa_gqa(self):
        original = torch.nn.functional.scaled_dot_product_attention
        with patch('torch.nn.functional.scaled_dot_product_attention', wraps=original) as attention:
            sdpa_repeat_kv(self.module, self.query, self.key, self.value, None)
        self.assertNotIn('enable_gqa', attention.call_args.kwargs)
        self.assertEqual(attention.call_args.args[1].shape[1], self.query.shape[1])

    def test_registers_the_standard_mask_factory(self):
        self.assertEqual(register_attention(), ATTENTION_IMPLEMENTATION)
        self.assertIs(AttentionMaskInterface()[ATTENTION_IMPLEMENTATION], AttentionMaskInterface()['sdpa'])

    def test_tiny_phi_logits_match_native_sdpa_with_padding_and_causality(self):
        register_attention()
        config = Phi3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, original_max_position_embeddings=64,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config._attn_implementation = 'sdpa'
        native = Phi3ForCausalLM(config).eval()
        custom_config = copy.deepcopy(config)
        custom_config._attn_implementation = ATTENTION_IMPLEMENTATION
        custom = Phi3ForCausalLM(custom_config).eval()
        custom.load_state_dict(native.state_dict())
        inputs = torch.tensor([[0, 0, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8]])
        mask = (inputs != 0).long()
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            expected = native(inputs, attention_mask=mask, position_ids=positions, use_cache=False).logits
            actual = custom(inputs, attention_mask=mask, position_ids=positions, use_cache=False).logits
            changed = inputs.clone()
            changed[:, -1] = 9
            changed_logits = custom(changed, attention_mask=mask, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[:, :-1], changed_logits[:, :-1], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
