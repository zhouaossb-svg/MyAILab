"""Command-line entry for single-image and batch MNIST digit recognition."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from model_utils import load_model, predict_from_pil, predict_images_in_folder


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "mnist_cnn_model.pth"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST 手写数字识别：单张图片或批量文件夹。",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path",
        type=Path,
        metavar="FILE",
        help="单张图片路径",
    )
    group.add_argument(
        "--batch",
        type=Path,
        metavar="DIR",
        help="批量识别：包含图片的文件夹路径",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"模型权重路径（默认: {_default_model_path()}）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="批量模式下将结果以 JSON 打印到标准输出",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model if args.model is not None else _default_model_path()

    try:
        model = load_model(model_path, device)
    except Exception as exc:
        print(f"模型加载失败: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.path is not None:
        image_path = args.path
        if not image_path.is_file():
            print(f"文件不存在或不是文件: {image_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with Image.open(image_path) as img:
                digit, confidence = predict_from_pil(img, model, device)
        except Exception as exc:
            print(f"识别失败: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"文件: {image_path.name}")
        print(f"预测数字: {digit}")
        print(f"置信度: {confidence * 100:.2f}%")
        return

    # --batch
    folder = args.batch
    try:
        results = predict_images_in_folder(folder, model, device)
    except Exception as exc:
        print(f"批量识别失败: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for row in results:
            if row["error"]:
                print(f"{row['filename']}: 错误 — {row['error']}")
            else:
                print(
                    f"{row['filename']}: {row['prediction']} "
                    f"(置信度 {row['confidence'] * 100:.2f}%)"
                )


if __name__ == "__main__":
    main()
