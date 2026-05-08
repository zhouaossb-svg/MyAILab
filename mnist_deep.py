import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    model = DeepNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

    torch.save(model.state_dict(), "mnist_deep_model.pth")
    print("训练完成，模型已保存为 mnist_deep_model.pth")

    simple_param_count = count_params(SimpleNet())
    deep_param_count = count_params(DeepNet())
    ratio = deep_param_count / simple_param_count
    print(
        f"参数量对比：SimpleNet = {simple_param_count}, "
        f"DeepNet = {deep_param_count}, 增加了 {ratio:.2f} 倍"
    )


if __name__ == "__main__":
    main()
