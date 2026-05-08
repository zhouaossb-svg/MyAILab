import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义 MNIST_CNN 卷积神经网络结构（用于 28x28 灰度图像分类）
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)       # 输出层（10 个服饰类别）
        self.relu = nn.ReLU()

    def forward(self, x):
        # 两层卷积 + 池化
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # 展平后接全连接层
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 1) 检查设备：优先使用 GPU，没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

    # 2) 定义数据预处理：把图片转成 Tensor，并做标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3) 自动下载 FashionMNIST 训练集
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # 4) 使用 DataLoader 按批次加载数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # 5) 创建 MNIST_CNN 模型，并放到对应设备上
    model = MNIST_CNN().to(device)

    # 6) 定义损失函数（Cost Function）和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7) 训练 2 轮（Epoch），并实时打印 Loss
    epochs = 2
    print("开始训练 Fashion-MNIST 模型...")
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            # 清空上一轮梯度
            optimizer.zero_grad()
            # 前向传播：得到模型输出
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播：根据损失计算梯度
            loss.backward()
            # 参数更新：按梯度更新权重
            optimizer.step()

            # 每 100 个 batch 打印一次 loss（实时观察训练变化）
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

    # 8) 训练完成后保存模型参数
    torch.save(model.state_dict(), "fashion_cnn_model.pth")
    print("Fashion-MNIST 模型训练完成，模型已保存为 fashion_cnn_model.pth")


if __name__ == "__main__":
    main()
