import torch
import math

torch.manual_seed(42)
x = torch.rand(1, 5, 8)

qkv_proj = torch.nn.Linear(8, 24)
qry, key, val = qkv_proj(x).split(8, dim=-1)

# Given the embedding of size 8, we want
# to split it and send it to different
# heads to compute a "partial" attention.
#
# Practically, the starting point is:
#
# qry -> (1, 5, 8)
# key -> (1, 5, 8)
# val -> (1, 5, 8)
#
# For each token we enhanced the embedding
# with the Linear layer. Now we want to split
# it again, 4 elements to each head.
#
# qry -> (1, 5, 2, 4)
# key -> (1, 5, 2, 4)
# val -> (1, 5, 2, 4)
#
# Multiplying the queries with the keys in
# this shape directly would not make sense.
# Stop and think about it, you will notice
# that we would multiply the same token query
# with the same token key. See the output:
#
# (1, 5, 2, 4) @ (1, 5, 4, 2) -> (1, 5, 2, 2)
#
# It could be read as: For each token we have
# an attention matrix of shape (2, 2) as the
# result of ones token query multiplied by the
# same token key.
#
# What we want instead is: For each head there
# is an attention matrix between the tokens with
# respective queries and keys.
#
# (1, ?, ?, ?) @ (1, ?, ?, ?) -> (1, 2, 5, 5)
#
# The aim is to group all the token queries
# for each head, then do the same for the keys.
#
# qry -> (1, 5, 2, 4) -> (1, 2, 5, 4)
# key -> (1, 5, 2, 4) -> (1, 2, 5, 4)
#
# To phrase it out loud: For each head group
# all the tokens queries and keys. Computing
# attention now is straight forward:
#
# qry @ key.transpose(-1, -2) -> (1, 2, 5, 5)
#
# From here on is the same as normal attention,
# with some minor changes to accommodate for the
# value shape concatenation.

qry = qry.reshape(1, 5, 2, 4).transpose(1, 2)      # (1, 5, 8) -> (1, 5, 2, 4) -> (1, 2, 5, 4)
key = key.reshape(1, 5, 2, 4).transpose(1, 2)      # (1, 5, 8) -> (1, 5, 2, 4) -> (1, 2, 5, 4)
val = val.reshape(1, 5, 2, 4).transpose(1, 2)      # (1, 5, 8) -> (1, 5, 2, 4) -> (1, 2, 5, 4)

att = qry @ key.transpose(-1, -2) / math.sqrt(4)   # (1, 2, 5, 5)
out = (att @ val).transpose(1, 2).reshape(1, 5, 8) # (1, 2, 5, 4) -> (1, 5, 2, 4) -> (1, 5, 8)
