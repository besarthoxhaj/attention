import matplotlib.pyplot as plt
import torch
import math

torch.manual_seed(42)
x = torch.rand(1, 5, 8)

qkv_proj = torch.nn.Linear(8, 24)

qry, key, val = qkv_proj(x).split(8, dim=-1)

# qkv -> (1, 5, 8 * 3) -> (1, 5, 24)
# qry -> qkv.split(..) -> (1, 5,  8)
# key -> qkv.split(..) -> (1, 5,  8)
# val -> qkv.split(..) -> (1, 5,  8)

att = qry @ key.transpose(-1, -2) / math.sqrt(8)
msk = torch.tril(torch.ones(5, 5)) == 0
att = att.masked_fill(msk, -float('inf'))
att = torch.softmax(att, dim=-1)
out = att @ val

fig = plt.figure(figsize=(10, 10))
num = att.squeeze().detach().numpy()
plt.matshow(num, cmap='viridis')
plt.colorbar()
plt.show()