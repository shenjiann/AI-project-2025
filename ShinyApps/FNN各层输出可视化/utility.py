from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "www" / "MNIST_samples"
MODELS_DIR = APP_DIR / "models"

def load_weights():
    # 读取权重和偏置文件
    fc1_w = pd.read_csv(MODELS_DIR / "weight_fc1.weight.csv", index_col=0).to_numpy(dtype=float).T
    fc1_b = pd.read_csv(MODELS_DIR / "weight_fc1.bias.csv", index_col=0).to_numpy(dtype=float).ravel()

    fc2_w = pd.read_csv(MODELS_DIR / "weight_fc2.weight.csv", index_col=0).to_numpy(dtype=float).T
    fc2_b = pd.read_csv(MODELS_DIR / "weight_fc2.bias.csv", index_col=0).to_numpy(dtype=float).ravel()

    fc3_w = pd.read_csv(MODELS_DIR / "weight_fc3.weight.csv", index_col=0).to_numpy(dtype=float).T
    fc3_b = pd.read_csv(MODELS_DIR / "weight_fc3.bias.csv", index_col=0).to_numpy(dtype=float).ravel()

    return (fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)

weights = load_weights()

def forward_pass(x_vec, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    """
    x_vec: (784,) 输入层（按行优先从 28x28 展平）
    返回：x, h1, h2, y 四个向量
    """
    def relu(x):
        return np.maximum(x, 0)

    def softmax(x):
        z = x - np.max(x)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    h1_pre = fc1_w @ x_vec + fc1_b
    h1 = relu(h1_pre)

    h2_pre = fc2_w @ h1 + fc2_b
    h2 = relu(h2_pre)

    y = fc3_w @ h2 + fc3_b
    y = softmax(y)

    return x_vec, h1, h2, y

def read_selected_image_to_vec(
    digit: str,
    idx: str | int,
) -> np.ndarray:
    """
    读取 www/MNIST_samples 下选择的图片，转为 (784,) 的 numpy 向量。
    """
    k = int(idx) - 1
    img_path = SAMPLES_DIR / f"{digit}_{k}.png"

    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) 
    x_vec = arr.flatten(order="C")
    x_vec /= 255.0
    return x_vec

import numpy as np

def vector_to_html_dots(vec, width_px=14, gap_px=4, pad_px=4):
    """
    将向量渲染为一排黑白圆点，灰度表示数值大小。
    白=最小值，黑=最大值。
    """
    v = np.asarray(vec, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return "<div class='dot-row'>无有效数据</div>"

    # 归一化到 [0,1]
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if np.isclose(vmin, vmax):
        norm = np.zeros_like(v)
    else:
        norm = (v - vmin) / (vmax - vmin)

    dots = []
    for xi, ni in zip(v, norm):
        gray = int(255 * (1 - ni))
        color = f"rgb({gray},{gray},{gray})"
        title = f"value={xi:.4g}"
        dots.append(
            f"<span class='dot' title='{title}' "
            f"style='background:{color}; width:{width_px}px; height:{width_px}px;'></span>"
        )

    html = (
        f"<div class='dot-row' "
        f"style='--gap:{gap_px}px; --pad:{pad_px}px; --dot:{width_px}px;'>"
        + "".join(dots) +
        "</div>"
    )
    return html

