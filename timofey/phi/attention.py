"""SDPA with explicit KV heads, including GPUs without Flash Attention GQA.

Transformers 4.57 may request ``enable_gqa=True`` for an unmasked CUDA
sequence. On V100 this selects the quadratic-memory math implementation.
Expanding KV heads first permits PyTorch's memory-efficient SDPA kernel.
The registered mask factory is the standard Transformers SDPA factory.
"""

from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface


ATTENTION_IMPLEMENTATION = "phi_sdpa_repeat_kv"


def sdpa_repeat_kv(module, query, key, value, attention_mask, dropout=0.0,
                   scaling=None, is_causal=None, **kwargs):
    if kwargs.get("output_attentions") or kwargs.get("head_mask") is not None:
        raise ValueError("This SDPA implementation does not return attention weights or apply head masks")
    query_heads, key_heads = query.shape[1], key.shape[1]
    if value.shape[1] != key_heads or query_heads % key_heads:
        raise ValueError("Attention head counts do not form complete KV groups")
    groups = query_heads // key_heads
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if attention_mask is not None:
        if attention_mask.ndim != 4:
            raise ValueError("The registered SDPA mask factory must provide a 4D mask or None")
        attention_mask = attention_mask[..., :key.shape[-2]]
    if is_causal is None:
        is_causal = bool(query.shape[-2] > 1 and attention_mask is None
                         and getattr(module, "is_causal", True))
    backend = nullcontext()
    if (query.is_cuda and query.shape[-2] > 4096
            and torch.cuda.get_device_capability(query.device)[0] < 8):
        # Fail explicitly if the efficient kernel is unavailable. Falling back to
        # a full attention matrix would exhaust V100 memory on few-shot prompts.
        backend = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    with backend:
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout,
            scale=scaling, is_causal=bool(is_causal))
    return output.transpose(1, 2).contiguous(), None


def register_attention():
    AttentionInterface.register(ATTENTION_IMPLEMENTATION, sdpa_repeat_kv)
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, AttentionMaskInterface()["sdpa"])
    return ATTENTION_IMPLEMENTATION
