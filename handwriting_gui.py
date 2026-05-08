import json
from pathlib import Path
from typing import Optional

import tkinter as tk
import torch
from PIL import Image, ImageDraw
from tkinter import messagebox

from model_utils import load_model, predict_from_pil, predict_images_in_folder


class HandwritingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST 手写数字识别")

        self.canvas_size = 280
        self.brush_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = Path(__file__).resolve().parent / "mnist_cnn_model.pth"
        try:
            self.model = load_model(model_path, self.device)
        except Exception as exc:
            messagebox.showerror("模型加载失败", f"无法加载模型:\n{exc}")
            raise

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.pack(padx=12, pady=12)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=4)

        predict_btn = tk.Button(
            control_frame,
            text="识别",
            width=10,
            command=self.predict,
        )
        predict_btn.pack(side=tk.LEFT, padx=6)

        clear_btn = tk.Button(
            control_frame,
            text="清空",
            width=10,
            command=self.clear_canvas,
        )
        clear_btn.pack(side=tk.LEFT, padx=6)

        self.result_var = tk.StringVar(value="请在黑色画布上写一个数字，然后点击“识别”。")
        result_label = tk.Label(root, textvariable=self.result_var, font=("Arial", 12))
        result_label.pack(pady=(8, 12))

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event: tk.Event) -> None:
        self.last_x, self.last_y = event.x, event.y
        self._draw_point(event.x, event.y)

    def on_paint(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                fill="white",
                width=self.brush_size,
                capstyle=tk.ROUND,
                smooth=True,
            )
            self.draw.line(
                [(self.last_x, self.last_y), (x, y)],
                fill=255,
                width=self.brush_size,
            )
        self.last_x, self.last_y = x, y

    def on_button_release(self, _event: tk.Event) -> None:
        self.last_x, self.last_y = None, None

    def _draw_point(self, x: int, y: int) -> None:
        radius = self.brush_size // 2
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("画布已清空，请重新书写数字。")

    def predict(self) -> None:
        try:
            digit, confidence = predict_from_pil(self.image, self.model, self.device)
            self.result_var.set(f"识别结果: {digit}    置信度: {confidence * 100:.2f}%")
        except Exception as exc:
            self.result_var.set("识别失败，请重试。")
            messagebox.showwarning("识别失败", str(exc))


def batch_predict_to_json(folder_path: str) -> str:
    """
    Convenience function for script usage:
    Accept a folder path and return prediction results JSON string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).resolve().parent / "mnist_cnn_model.pth"
    model = load_model(model_path, device)
    results = predict_images_in_folder(folder_path, model, device)
    return json.dumps(results, ensure_ascii=False, indent=2)


def main() -> None:
    root = tk.Tk()
    _app = HandwritingApp(root)
    root.resizable(False, False)
    root.mainloop()


if __name__ == "__main__":
    main()
