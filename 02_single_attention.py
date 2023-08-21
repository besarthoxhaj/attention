import torch

torch.manual_seed(42)
x = torch.rand(1, 5, 8)

qkv_proj = torch.nn.Linear(8, 24)
qry, key, val = qkv_proj(x).split(8, dim=-1)
out = torch.nn.functional.scaled_dot_product_attention(qry, key, val, is_causal=True)