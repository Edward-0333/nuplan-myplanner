import torch

# 假设你有一个批次，其中每个场景的转移代价矩阵大小不同
transition_costs = [torch.randn(k, k) for k in [3, 5, 4]]  # 示例，k分别是3, 5, 4

# 计算批次中最大的k值
k_max = max(k for k in [3, 5, 4])


# 填充转移代价矩阵，使其统一大小
def pad_transition_costs(transition_costs, k_max):
    padded_costs = []
    for M in transition_costs:
        k = M.size(0)
        # 计算填充的大小
        pad_size = (0, k_max - k, 0, k_max - k)  # pad后两个维度（右侧和下方）
        # 填充矩阵，填充值使用 float('-inf') 来避免对计算产生影响
        padded_M = torch.nn.functional.pad(M, pad_size, value=float('-inf'))
        padded_costs.append(padded_M)

    return padded_costs


# 填充后的转移代价矩阵
padded_transition_costs = pad_transition_costs(transition_costs, k_max)

# 将填充后的转移代价矩阵堆叠成一个 [B, k_max, k_max] 的张量
batch_transition_costs = torch.stack(padded_transition_costs)  # [B, k_max, k_max]

# 假设批次大小 B = 3，车道数 N = 5，时间步 T = 10
B, N, T, K = 3, 5, 10, k_max  # 示例：3个场景，5个车，10个时间步
p_t = torch.randn(B, N, T - 1, K)  # [B, N, T-1, K]
p_tp1 = torch.randn(B, N, T - 1, K)  # [B, N, T-1, K]

# 检查矩阵维度是否匹配
assert p_t.shape[-1] == batch_transition_costs.shape[-1], "p_t 的最后一维与转移代价矩阵的车道数不匹配"

# 计算从 t 到 t+1 的转移代价时的矩阵乘法
# 计算 Mp_tp1 时需要转置 batch_transition_costs 以便匹配维度
# (B, N, T-1, K) @ (B, K_max, K_max) -> [B, N, T-1, K_max]
Mp_tp1 = torch.matmul(p_tp1, batch_transition_costs.transpose(-1, -2))  # [B, N, T-1, K_max]

# p_t^T * Mp_tp1，按 K_max 的维度求和
L_geo_trans = (p_t * Mp_tp1).sum(-1)  # [B, N, T-1]

# 按有效位置进行归一化
valid_t = torch.ones_like(L_geo_trans)  # 假设全是有效位置
L_geo_trans = (L_geo_trans * valid_t).sum() / valid_t.sum().clamp_min(1.0)

# 打印结果
print(L_geo_trans)
