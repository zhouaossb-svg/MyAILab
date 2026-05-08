import torch

# 第一步：创建两个随机矩阵 A 和 B（这里都用 3x3，方便查看）
A = torch.rand(3, 3)
B = torch.rand(3, 3)

# 第二步：让矩阵 A 和 B 相乘（矩阵乘法，不是逐元素相乘）
C = torch.matmul(A, B)

# 第三步：检查当前环境是否有可用的 CUDA（NVIDIA GPU）
cuda_available = torch.cuda.is_available()

# 第四步：把结果打印出来，方便你确认环境是否正常
print("矩阵 A：")
print(A)
print("\n矩阵 B：")
print(B)
print("\nA × B 的结果 C：")
print(C)

print("\nCUDA 是否可用：", cuda_available)
if cuda_available:
    # 如果有 GPU，再打印 GPU 名称
    print("当前可用 GPU：", torch.cuda.get_device_name(0))
else:
    print("当前使用 CPU（未检测到可用 CUDA GPU）")
