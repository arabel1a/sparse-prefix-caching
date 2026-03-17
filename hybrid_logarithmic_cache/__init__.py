import torch
import torch.nn.functional as F
import time
import math
from transformers import AutoModelForCausalLM, AutoConfig
import logging


logging.warning("Monkey-patching SPDA for Group Query Attention + transformers!!")

# This patch automatically expands K and V to match Q for GQA models.
_original_sdpa = F.scaled_dot_product_attention

def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=None):
    bq, hq, sq, dq = query.shape
    bk, hk, sk, dk = key.shape
    if hq != hk:
        num_rep = hq // hk
        key = key.unsqueeze(2).expand(bk, hk, num_rep, sk, dk).reshape(bk, hq, sk, dk)
        value = value.unsqueeze(2).expand(bk, hk, num_rep, sk, dk).reshape(bk, hq, sk, dk)

    return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=None, enable_gqa=False)

F.scaled_dot_product_attention = _patched_sdpa
