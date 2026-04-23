from .patches import apply_sdpa_patch, apply_patched_gdn_forward, enforce_efficient_attention

enforce_efficient_attention()
apply_sdpa_patch()
apply_patched_gdn_forward()

