import torch
from .patches import apply_sdpa_patch, apply_patched_gdn_forward, enforce_efficient_attention

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    enforce_efficient_attention()
    apply_sdpa_patch()
apply_patched_gdn_forward()

