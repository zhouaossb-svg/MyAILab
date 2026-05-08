import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from mnist_cnn import MNIST_CNN  # 导入你的模型结构

# 1. 加载你训练好的“学霸”大脑
model = MNIST_CNN()
model.load_state_dict(torch.load('fashion_cnn_model.pth'))
model.eval()

# 2. 准备数据：从 MNIST 测试集中拿第一张图片
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
image, label = test_data[0]  # image 形状是 [1, 28, 28]

# 将图片增加一个 batch 维度，变成 [1, 1, 28, 28]，准备送入模型
image_tensor = image.unsqueeze(0)

# 3. 核心截获：手动让图片只通过第一层卷积
with torch.no_grad():
    # 我们不执行整个 forward，只提取 conv1 的输出
    features = model.conv1(image_tensor) # 形状会变成 [1, 32, 26, 26]

# 4. 把 32 张“滤镜”图画出来
fig, axes = plt.subplots(4, 8, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    # 取出第 i 个通道的特征图，转成 numpy 格式以便画图
    feature_map = features[0, i].numpy()
    ax.imshow(feature_map, cmap='viridis') # 使用 viridis 伪彩色，亮色代表特征激活强烈
    ax.axis('off')
    ax.set_title(f"Filter {i+1}", fontsize=10)
    # ------------------------------------
# 强制手动检查
# ------------------------------------
print(f"特征图的形状是: {features.shape}") # 应该是 [1, 32, 26, 26]
print("正在尝试把第一号滤镜看到的内容打印成数字矩阵...")

# 打印出第一个滤镜的前 5x5 个数字
print(features[0, 0, :5, :5])

plt.savefig('check_me.png')
print("\n--- 关键提示 ---")
print("如果终端输出了上面的数字矩阵，说明你的基础完全没问题，程序跑通了！")
print("请现在去左边找 check_me.png。")