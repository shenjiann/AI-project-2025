import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from models.AlexNet import AlexNet


APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "www" / "cifar10_samples"
MODELS_DIR = APP_DIR / "models"

model = AlexNet(num_classes=10)
state = torch.load(MODELS_DIR/'AlexNet.pth', map_location="cpu")
model.load_state_dict(state)
model.eval()
conv_layers = {
    "conv1": model.features[0],
    "conv2": model.features[4],
    "conv3": model.features[8],
    "conv4": model.features[10],
    "conv5": model.features[12],
}

def plot_weight_hist(w, bins=60, xlim=None):
    w = np.asarray(w).ravel()
    mu, sd = float(np.mean(w)), float(np.std(w))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist(w, bins=bins, histtype="stepfilled",
            edgecolor="black", linewidth=1.0,
            facecolor="0.85", alpha=1.0)
    ax.axvline(0, color="0.25", lw=0.8, ls="--")
    ax.text(0.98, 0.95, f"μ = {mu:.3g} \n σ = {sd:.3g}", transform=ax.transAxes,
            ha="right", va="top")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_xlim(-0.25, 0.25)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    return fig


def pick_conv_weights(which_key: str):
    """
    从已加载的模型中取出对应卷积层权重 (numpy array) 和用于标题的文本。
    依赖你之前建立的 `_conv_layers = {"conv1": ..., ..., "conv5": ...}`
    """
    layer = conv_layers.get(which_key, None)
    if layer is None or not hasattr(layer, "weight"):
        return None, f"{which_key.upper()} weights not found"

    with torch.no_grad():
        w = layer.weight.detach().cpu().numpy()
    return w
