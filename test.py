import torch
import torch.nn as nn

# 输入的维度
embed_dim = 256  # 每个 token 的嵌入维度
num_heads = 8  # 多头的数量
batch_size = 32  # 批量大小
seq_len = 10  # 序列长度

# 创建 MultiheadAttention 层
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

# 输入查询（query）、键（key）和值（value）
query = torch.rand(seq_len, batch_size, embed_dim)  # 形状 (seq_len, batch_size, embed_dim)
key = torch.rand(seq_len, batch_size, embed_dim)  # 形状 (seq_len, batch_size, embed_dim)
value = torch.rand(seq_len, batch_size, embed_dim)  # 形状 (seq_len, batch_size, embed_dim)

# 使用 MultiheadAttention
attn_output, attn_output_weights = multihead_attn(query, key, value)

print("Attention Output Shape:", attn_output.shape)  # (seq_len, batch_size, embed_dim)
print("Attention Weights Shape:", attn_output_weights.shape)  # (batch_size, num_heads, seq_len, seq_len)
