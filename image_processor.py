"""OpenCV-based preprocessing for MNIST-style digit images.

Pipeline:
  1. Grayscale + Otsu threshold → binary
  2. Ensure white digit on black background
  3. Find largest contour → crop to bounding box
  4. Square + pad → resize to 20×20 (aspect-ratio-preserved)
  5. Center on 28×28 canvas → normalize
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class RecognitionError(Exception):
    """Raised when image preprocessing fails or related validation fails."""


def _ensure_digit_white_on_black(binary: np.ndarray) -> np.ndarray:
    """Invert a binary image so the digit is white (255) on black (0)."""
    if np.count_nonzero(binary == 255) > np.count_nonzero(binary == 0):
        binary = cv2.bitwise_not(binary)
    return binary


def _crop_to_digit(binary: np.ndarray) -> np.ndarray:
    """Find the largest contour, crop to bounding box, and return a
    28×28 centered image.  Falls back to a centered 28×28 resize when
    no contour is found."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Empty canvas — return a blank 28×28
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = binary[y:y + h, x:x + w]

    # Pad the shorter side to make a square
    size = max(w, h)
    t = (size - h) // 2
    b = size - h - t
    l = (size - w) // 2
    r = size - w - l
    square = cv2.copyMakeBorder(cropped, t, b, l, r,
                                cv2.BORDER_CONSTANT, value=0)

    # Add proportional padding (~50 % of digit size) so the digit
    # doesn't touch the edges after the final resize
    pad = size // 2
    padded = cv2.copyMakeBorder(square, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)

    # Resize digit to 20×20 (aspect ratio preserved by the square + pad)
    resized_20 = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)

    # Center on 28×28 canvas (4 px margin on each side)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized_20
    return canvas


def preprocess_image_opencv(img: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image with OpenCV and output MNIST-shaped tensor.

    Uses contour cropping: detects the digit, crops to bounding box,
    scales while preserving aspect ratio, and centers on 28×28.

    Returns:
        torch.Tensor: Shape [1, 1, 28, 28].
    """
    if img is None:
        raise RecognitionError("输入图片为空。")

    try:
        gray = np.array(img.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise RecognitionError(f"图片灰度化失败: {exc}") from exc

    try:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception as exc:
        raise RecognitionError(f"OpenCV 二值化失败: {exc}") from exc

    binary = _ensure_digit_white_on_black(binary)

    try:
        canvas = _crop_to_digit(binary)
    except Exception as exc:
        raise RecognitionError(f"轮廓裁剪失败: {exc}") from exc

    tensor = transforms.ToTensor()(canvas)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    return tensor.unsqueeze(0)
