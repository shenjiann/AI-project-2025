from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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


def vector_to_html_dots(vec, width_px=14, gap_px=4, pad_px=4, scroll=False):
    """
    将向量渲染为黑白圆点。
    - scroll=False: 用 .dot-row（可换行）
    - scroll=True : 用 .dot-strip（单行，支持横向滚动）
    """
    v = np.asarray(vec, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return "<div class='dot-row'>无有效数据</div>"

    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if np.isclose(vmin, vmax):
        norm = np.zeros_like(v)
    else:
        norm = (v - vmin) / (vmax - vmin)

    dots = []
    for xi, ni in zip(v, norm):
        gray = int(255 * (1 - float(ni)))  # 小值白，大值黑
        color = f"rgb({gray},{gray},{gray})"
        title = f"value={xi:.4g}"
        dots.append(
            f"<span class='dot' title='{title}' "
            f"style='background:{color}; width:{width_px}px; height:{width_px}px;'></span>"
        )

    container_cls = "dot-strip" if scroll else "dot-row"
    html = (
        f"<div class='{container_cls}' "
        f"style='--gap:{gap_px}px; --pad:{pad_px}px; --dot:{width_px}px;'>"
        + "".join(dots) +
        "</div>"
    )
    return html


def plot_weight_hist(w, title="Weights Histogram", bins=60):
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    ax.hist(w.ravel(), bins=bins, alpha=0.8, color="steelblue", label=f"W {tuple(w.shape)}")
    ax.set_title(title)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.set_xlim((-6, 4))
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    all_weights = []
    try:
        all_weights.append(weights[0].ravel())
        all_weights.append(weights[2].ravel())
        all_weights.append(weights[4].ravel())
    except Exception:
        pass

    if all_weights:
        global_min = min(w.min() for w in all_weights)
        global_max = max(w.max() for w in all_weights)
    else:
        global_min, global_max = -1, 1
    
    print(f"All weights value range: [{global_min:.4g}, {global_max:.4g}]")