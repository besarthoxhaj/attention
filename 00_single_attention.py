import matplotlib.pyplot as plt
import torch
import math

torch.manual_seed(42)
x = torch.rand(1, 5, 8)

qry_proj = torch.nn.Linear(8, 8)
key_proj = torch.nn.Linear(8, 8)
val_proj = torch.nn.Linear(8, 8)

qry = qry_proj(x)
key = key_proj(x)
val = val_proj(x)

# qry -> (1, 5, 8)
# key -> (1, 5, 8)
# val -> (1, 5, 8)

# att -> (1, 5, 8) @ (1, 8, 5) = (1, 5, 5)
# out -> (1, 5, 5) @ (1, 5, 8) = (1, 5, 8)

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