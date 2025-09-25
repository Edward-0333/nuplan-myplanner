import torch
import torch.nn as nn

# 定义 MultiheadAttention
mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

batch_size = 1
len_q = 2   # Q 的长度
len_kv = 4  # K 和 V 的长度（比 Q 长）

# 构造 Q, K, V
Q = torch.randn(batch_size, len_q, 8)   # [1, 2, 8]
K = torch.randn(batch_size, len_kv, 8)  # [1, 4, 8]
V = torch.randn(batch_size, len_kv, 8)  # [1, 4, 8]

# key_padding_mask: [batch_size, len_kv]
# True 表示该位置需要被 mask 掉（即忽略）
key_padding_mask = torch.tensor([[False, False, True, True]])  # 屏蔽掉后两个位置

# 运行 attention
output, attn_weights = mha(Q, K, V, key_padding_mask=key_padding_mask)

print("输出 shape:", output.shape)          # [1, 2, 8]
print("注意力权重 shape:", attn_weights.shape)  # [1, 2, 4]
print("注意力权重：")
print(attn_weights)
