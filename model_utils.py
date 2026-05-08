"""MNIST CNN model definition, loading, and inference helpers."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from PIL import Image

from image_processor import RecognitionError, preprocess_image_opencv


class MNIST_CNN(nn.Module):
    """CNN for 28x28 single-channel digit classification (matches mnist_cnn_model.pth)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(
    model_path: Path,
    device: torch.device,
    model_class: Type[torch.nn.Module] = MNIST_CNN,
) -> torch.nn.Module:
    """Load model weights safely and switch to eval mode."""
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_model_inference(
    model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> Tuple[int, float]:
    """
    Run model inference and return predicted digit and confidence.

    Returns:
        Tuple[int, float]: (digit, confidence in range [0, 1]).
    """
    if image_tensor.ndim != 4:
        raise RecognitionError("模型输入维度错误，期望 [N, C, H, W]。")

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return int(pred.item()), float(confidence.item())


def predict_from_pil(
    img: Image.Image, model: torch.nn.Module, device: torch.device
) -> Tuple[int, float]:
    """Pipeline: OpenCV preprocessing + PyTorch inference."""
    tensor = preprocess_image_opencv(img)
    return run_model_inference(model, tensor, device)


def predict_images_in_folder(
    folder_path: Union[str, Path], model: torch.nn.Module, device: torch.device
) -> List[Dict[str, Any]]:
    """
    Batch predict all images in a folder.

    Returns:
        JSON-serializable records:
        {"filename", "prediction", "confidence", "error"}
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是目录: {folder}")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    results: List[Dict[str, Any]] = []
    for file_path in sorted(files):
        item: Dict[str, Any] = {
            "filename": file_path.name,
            "prediction": None,
            "confidence": None,
            "error": None,
        }
        try:
            with Image.open(file_path) as img:
                digit, conf = predict_from_pil(img, model, device)
            item["prediction"] = digit
            item["confidence"] = round(conf, 6)
        except Exception as exc:
            item["error"] = f"识别失败: {exc}"
        results.append(item)

    return results
