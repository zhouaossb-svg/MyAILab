import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from mnist_cnn import MNIST_CNN


def preprocess_image(image_path: str) -> torch.Tensor:
    # 1) 灰度读取
    arr = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    blur = cv2.GaussianBlur(arr, (5, 5), 0)

    # 2) 自适应二值化（保留细节）+ OTSU（二值稳定）
    # 两种方式都尝试，后面自动选择笔画更完整的一张。
    _, binary_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2,
    )

    def score_foreground(binary_img: np.ndarray) -> float:
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        # 更偏向“主轮廓面积更大”的结果，避免笔迹被洗掉
        return float(cv2.contourArea(largest))

    binary = binary_adapt if score_foreground(binary_adapt) >= score_foreground(binary_otsu) else binary_otsu

    # 自动保证“黑底白字”
    if int((binary == 255).sum()) > binary.size // 2:
        binary = 255 - binary

    # 3) 轻量闭运算连接断裂笔画，尽量保留“5”的连贯轮廓
    close_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 4) 查找轮廓并裁剪数字边界
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cropped = np.zeros((20, 20), dtype=np.uint8)
    else:
        # 选面积最大的轮廓，避免小噪声干扰
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = binary[y : y + h, x : x + w]

    # 5) 笔画加粗（Dilation）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cropped, kernel, iterations=2)

    # 6) 缩放到 20x20（保持长宽比），再居中放到 28x28 黑底
    target_size = 20
    h, w = dilated.shape
    if h == 0 or w == 0:
        resized = np.zeros((target_size, target_size), dtype=np.uint8)
    elif w > h:
        new_w = target_size
        new_h = max(1, int(round(h * target_size / w)))
        resized = cv2.resize(dilated, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        new_h = target_size
        new_w = max(1, int(round(w * target_size / h)))
        resized = cv2.resize(dilated, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    final_arr = np.zeros((28, 28), dtype=np.uint8)
    offset_x = (28 - resized.shape[1]) // 2
    offset_y = (28 - resized.shape[0]) // 2
    final_arr[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    final_img = Image.fromarray(final_arr, mode="L")

    # 7) 保存证据图
    final_img.save("final_input.png")

    # 8) To tensor + normalization used during training
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tensor = transform(final_img).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor


def main():
    image_path = "my_number.jpg"
    model_path = "mnist_cnn_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MNIST_CNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocess image and run inference
    x = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_digit = pred.item()
    confidence_pct = confidence.item() * 100

    print(f"AI 预测结果: {pred_digit}")
    print(f"置信度: {confidence_pct:.2f}%")


if __name__ == "__main__":
    main()
