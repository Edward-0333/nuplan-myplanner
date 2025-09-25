import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

Q = torch.randn(1, 2, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)

# attn_mask: 屏蔽掉第一个 query 的所有 key
attn_mask = torch.zeros(2, 4)
attn_mask[0, :] = float('-inf')

output, attn_weights = mha(Q, K, V, attn_mask=attn_mask)

# 需要忽略的 query（例如第 0 个 query）
ignore_query = torch.tensor([True, False], dtype=torch.bool)

# 在输出阶段把对应的结果清零/忽略
output = output.clone()  # 克隆以免原地修改带来梯度警告
output.masked_fill_(ignore_query.view(1, -1, 1), 0.0)

# 注意力权重包含 NaN 时可以先转为 0，再做同样的屏蔽
attn_weights = torch.nan_to_num(attn_weights)
attn_weights = attn_weights.clone()
attn_weights.masked_fill_(ignore_query.view(1, -1, 1), 0.0)

print("忽略后的输出：")
print(output)
print("注意力权重：")
print(attn_weights)
print("每行和：", attn_weights.sum(-1))
