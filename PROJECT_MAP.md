# MyAILab 项目资产地图

> 基于 Python 3.7 + PyTorch 1.13 + OpenCV 4.13 的手写数字识别工程。
> 最后更新时间: 2026-05-08

---

## 一、目录结构

```
MyAILab/
│
├── # ──── 核心管线（推理入口） ────
├── main.py                 # CLI 入口：支持 --path（单张）和 --batch（批量）
├── model_utils.py          # 模型定义 MNIST_CNN + 加载/推理/批量预测
├── image_processor.py      # 预处理管线：Otsu → 轮廓裁剪 → 居中 → [1,1,28,28]
│
├── # ──── GUI 交互（独立入口） ────
├── handwriting_gui.py      # tkinter 手写画板，调用同一管线做实时识别
│
├── # ──── 训练脚本（生成 .pth） ────
├── mnist_cnn.py            # MNIST_CNN 训练 + 保存 mnist_cnn_model.pth（主力模型）
├── mnist_train.py          # 旧版 CNN（padding=1 版），保存 fashion_cnn_model.pth
├── mnist_deep.py           # DeepNet（5 层全连接），保存 mnist_deep_model.pth
│
├── # ──── 预训练权重（推理用） ────
├── mnist_cnn_model.pth     # [主力] MNIST_CNN · 参数量 225,034
├── mnist_deep_model.pth    # DeepNet 5 层 · 参数量 669,706
├── mnist_model.pth         # SimpleNet 3 层 · 参数量 109,386
├── fashion_cnn_model.pth   # FashionMNIST 版 CNN · 31,338 参数
│
├── # ──── 历史/工具脚本 ────
├── inference.py            # 旧版预处理（自适应阈值 + 闭运算 + 膨胀 + 居中）
├── hello_pytorch.py        # PyTorch 环境测试脚本（矩阵乘法）
├── visualize_features.py   # 可视化 CNN 第一层卷积核输出 → check_me.png
│
├── # ──── HTML 教学文档 ────
├── neuron.html             # 单个神经元可视化
├── backprop_viz.html       # 反向传播流程动画
├── gradient_descent.html   # 梯度下降 3D 演示
├── mnist_hero.html         # MNIST 全连接网络交互图
│
└── # ──── 测试 / 证据图片 ────
    ├── my_number.jpg        # 手写数字 3 照片
    ├── check_me.png         # visualize_features.py 生成的 CNN 特征图
    ├── my_cnn_vision.png    # 特征可视化全览
    ├── debug_input.png      # 预处理中间结果（旧版生成）
    └── final_input.png      # 预处理最终 28×28（旧版生成）
```

---

## 二、核心数据流

### 推理管线（本质是一条链）

```
┌──────────────┐     ┌─────────────────────────────┐     ┌──────────────────┐
│  输入来源    │     │   image_processor.py         │     │  model_utils.py  │
│              │     │                              │     │                  │
│  --path FILE │────→│  preprocess_image_opencv()   │────→│  run_model_      │
│  GUI 画布    │     │  ① 灰度化                     │     │  inference()     │
│  --batch DIR │     │  ② Otsu 二值化                 │     │  ① softmax       │
│              │     │  ③ 保证白字黑底                 │     │  ② argmax        │
│              │     │  ④ 最大轮廓检测 + 裁剪           │     │  ③ 返回(数字, %) │
│              │     │  ⑤ 方形 padding + 缩放 20×20   │     │                  │
│              │     │  ⑥ 居中到 28×28 画布            │     └──────────────────┘
│              │     │  ⑦ Normalize((0.1307,),        │           │
│              │     │       (0.3081,))               │           │
│              │     │  ⑧ unsqueeze → [1,1,28,28]    │           ▼
│              │     └─────────────────────────────┘     ┌──────────────────┐
│              │                                          │   输出           │
│              │                                          │ "预测数字: 3"    │
│              │                                          │ "置信度: 92.3%"  │
└──────────────┘                                          └──────────────────┘
```

### 各模块职责矩阵

| 模块 | 输入 | 输出 | 关键函数 |
|---|---|---|---|
| `main.py` | CLI 参数 | 终端打印 / JSON | `main()`, `_parse_args()` |
| `model_utils.py` | PIL Image + Model | `(int digit, float conf)` | `predict_from_pil()` |
| `image_processor.py` | PIL Image | `torch.Tensor [1,1,28,28]` | `preprocess_image_opencv()` |
| `handwriting_gui.py` | 鼠标绘制事件 | tkinter 弹窗 | `HandwritingApp.predict()` |

### 调用拓扑

```
main.py ──→ model_utils.predict_from_pil()
                ├── image_processor.preprocess_image_opencv()
                └── model_utils.run_model_inference()

handwriting_gui.py ──→ model_utils.predict_from_pil()
                           ├── image_processor.preprocess_image_opencv()
                           └── model_utils.run_model_inference()
```

CLI 和 GUI 共享完全相同的预处理 + 推理管线，确保结果一致。

---

## 三、关键重构决策

### 决策 1：从「直接缩放」→「轮廓裁剪」（2026-05-07）

| 维度 | 旧方案（v1） | 新方案（v2） |
|---|---|---|
| 核心操作 | `cv2.resize(binary, (28,28))` | 检测最大轮廓 → 裁剪 → 方形 padding → 缩放 20×20 → 居中到 28×28 |
| 位置鲁棒性 | ❌ 数字画在角落会被压扁 | ✅ 始终居中，位置无关 |
| 尺度鲁棒性 | ❌ 大小数字在 28×28 中占比差异大 | ✅ 统一缩放到固定比例 |
| MNIST 匹配度 | 中 | 高（标准 MNIST 预处理就是居中+留边） |

**为什么改：**

- 旧方案直接把整张图压成 28×28，若数字只占画布一小部分，压完后数字像素极度稀疏，模型几乎"看不见"有效笔画。
- 轮廓裁剪保证了数字始终填满 28×28 的有效区域（四周留 4px 白边），与训练时的 MNIST 数据分布一致，这对卷积网络提取特征至关重要。
- 对 GUI 场景尤为重要——用户在 280×280 画布上随手画一个小数字，旧版会被直接缩小成一团噪声，新版能准确定位并居中。

### 决策 2：去除 `inference.py` 中的「双阈值择优」逻辑

旧版 `inference.py` 同时计算 Otsu 阈值和自适应阈值，根据主轮廓面积选更优的一个。新版统一为 Otsu：

- Otsu 在 MNIST 场景下已足够稳定（手写数字是典型的双峰直方图）。
- 自适应阈值在均匀光照的 GUI 画布上没有额外优势，反而增加 1 倍计算量。
- 保持了管线简单可维护。

### 决策 3：模块解耦（`model_utils.py` + `image_processor.py`）

旧版 `inference.py` 将预处理、模型加载、推理全部揉在一个文件里。重构后：

- `image_processor.py` — 仅负责图像 → tensor 的转换，可独立替换预处理策略。
- `model_utils.py` — 仅负责模型管理和推理，不关心输入图片格式。
- `main.py` / `handwriting_gui.py` — 仅负责 I/O 交互，调用统一接口。

---

## 四、模型架构

### MNIST_CNN（主力模型，225K 参数）

```
Input [1, 28, 28]
    ↓ Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)    → [32, 13, 13]
    ↓ Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)   → [64, 5, 5]
    ↓ Flatten (1600)
    ↓ Linear(1600→128) + ReLU
    ↓ Linear(128→10)
    ↓ Softmax
Output: digit 0-9
```

对应 `mnist_cnn_model.pth`（8 个 state_dict 键，225,034 参数）。训练代码在 `mnist_cnn.py`。

### 其他模型

| 模型 | 文件 | 架构 | 参数量 | 用途 |
|---|---|---|---|---|
| SimpleNet | `mnist_model.pth` | 3 层全连接 | 109,386 | 入门基线 |
| DeepNet | `mnist_deep_model.pth` | 5 层全连接 | 669,706 | 深度对比 |
| FashionCNN | `fashion_cnn_model.pth` | CNN + padding=1 | 31,338 | FashionMNIST 分类 |

---

## 五、环境配置指南

### 运行时依赖

| 包 | 版本 | 用途 |
|---|---|---|
| Python | 3.7.8 | 运行时 |
| PyTorch | 1.13.1+cpu | 模型推理/训练 |
| torchvision | 0.14.1 | MNIST 数据集、图像变换 |
| OpenCV | 4.13.0 | 图像预处理（二值化、轮廓、缩放） |
| Pillow | 9.5.0 | PIL Image 加载/保存 |
| NumPy | 1.21.6 | 矩阵操作 |
| Matplotlib | 3.5.3 | 特征可视化 |
| tkinter | 内置 | GUI 手写画板 |

> **硬件加速：** 当前环境 `CUDA available: False`，全 CPU 推理。单张图片推理耗时约 < 50ms，完全可用。

### 安装命令

```bash
# 基础依赖（一行装完所有）
pip install torch==1.13.1 torchvision==0.14.1 opencv-python==4.13.0.92 pillow==9.5.0 numpy==1.21.6 matplotlib==3.5.3

# 验证安装
python -c "
import torch, cv2, numpy as np, PIL, matplotlib
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pillow: {PIL.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
"
```

### 网络/代理配置

本环境**无 HTTP 代理**。PyTorch 和 pip 直连下载。若身处受限网络：

```bash
# 为 pip 设置代理
pip install --proxy http://user:pass@host:port <package>

# 为 torchvision 下载 MNIST 数据集设置代理
import os
os.environ['HTTP_PROXY'] = 'http://user:pass@host:port'
os.environ['HTTPS_PROXY'] = 'http://user:pass@host:port'
```

### 快速启动

```bash
cd MyAILab

# 单张图片识别
python main.py --path my_number.jpg

# 批量识别目录下所有图片
python main.py --batch . --json

# 启动 GUI 手写板
python handwriting_gui.py

# 重新训练模型
python mnist_cnn.py
```

---

## 六、测试图片清单

| 文件名 | 像素 | 内容 | 测试用途 |
|---|---|---|---|
| `my_number.jpg` | 81242 B | 手写数字 | CLI 识别主测试 |
| `check_me.png` | 70557 B | CNN 特征可视化 | 视觉验证 |
| `my_cnn_vision.png` | 55214 B | 特征全览 | 教学演示 |
| `debug_input.png` | 96 B | 预处理中间态 | 调试 |
| `final_input.png` | 128 B | 28×28 最终输入 | 调试/验证 |
