"""
Backward-compatible entry: re-exports MNIST_CNN and provides MNIST training script.

The canonical model definition lives in model_utils.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_utils import MNIST_CNN

__all__ = ["MNIST_CNN", "count_params", "main"]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
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

    model = MNIST_CNN().to(device)
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

    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    print("训练完成，模型已保存为 mnist_cnn_model.pth")

    cnn_param_count = count_params(MNIST_CNN())
    deepnet_param_count = 570000
    ratio = cnn_param_count / deepnet_param_count
    print(f"MNIST_CNN 总参数量: {cnn_param_count}")
    print(
        f"与 DeepNet (约 {deepnet_param_count}) 对比："
        f"MNIST_CNN 是其 {ratio:.2f} 倍"
    )


if __name__ == "__main__":
    main()
