import numpy as np

# 假设最大目标数量
max_objects = 10

# 示例标签
targets = [
    [147, 174, 220, 222, 1],
    [157, 260, 213, 356, 2],
    [233, 260, 260, 308, 1]
]

# 填充标签
padded_targets = np.full((max_objects, 5), -1, dtype=np.int32)
for i, target in enumerate(targets):
    padded_targets[i] = target

print(padded_targets)
