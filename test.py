import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

A = torch.randn(1, 5, 3)
B = torch.randn(8, 5, 2)

# 目标特征维度
max_feat = max(A.shape[-1], B.shape[-1])

def pad_last_dim(x, target_dim):
    diff = target_dim - x.shape[-1]
    if diff > 0:
        # pad 格式是 (最后一维前, 最后一维后, 倒数第二维前, 倒数第二维后, ...)
        x = F.pad(x, (0, diff), mode='constant', value=0)
    return x

A_padded = pad_last_dim(A, max_feat)
B_padded = pad_last_dim(B, max_feat)
print(A_padded.shape, B_padded.shape)
