import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from models.AlexNet import AlexNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# 特征图提取
_to_tensor = T.Compose([
    T.Resize((227, 227)),
    T.ToTensor(),  # [0,1]
    T.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),  # CIFAR-10 常用
])

def load_sample_tensor(cls: str, k: int):
    """读取样本png -> 预处理 -> (1,3,32,32) tensor"""
    import PIL.Image as Image
    p = SAMPLES_DIR / f"{cls}_{k}.png"
    img = Image.open(p).convert("RGB")
    x = _to_tensor(img).unsqueeze(0).to(DEVICE)
    return x

# --- 前向钩子抓取各卷积层输出 ---
# 假设你的 AlexNet 里卷积层名依次为：features[0], [3], [6], [8], [10]
# 如果你自定义命名不同，请把 mapping 改成你模型的实际层。
def grab_conv_activations(model, x, layer_names=("conv1","conv2","conv3","conv4","conv5")):
    """
    返回: { 'conv1': (C,H,W) tensor(cpu), ... }  (已ReLU)
    - 兼容 out 是 tuple 的情况（比如返回 (x, indices)）
    - 兼容 out 仍为 (N,C,H,W) 的情况，统一取 batch 0
    - 前向时临时把所有 MaxPool2d 的 ceil_mode=True，避免 0×0
    """
    import torch.nn as nn
    import torch.nn.functional as F

    feats = {}
    handles = []
    patched_pools = []

    # === 按你的 AlexNet 实际结构调整 ===
    name_to_module = {
        "conv1": model.features[0],
        "conv2": model.features[3],
        "conv3": model.features[6],
        "conv4": model.features[8],
        "conv5": model.features[10],
    }

    def _mk_hook(key):
        def hook(m, inp, out):
            # 1) 先把 tuple/list 解成 Tensor
            if isinstance(out, (tuple, list)):
                out = out[0]
            # 2) 若仍是 NCHW，取 batch=0
            if out.dim() == 4:
                out = out[0]
            # 3) ReLU 后转 cpu，形状应为 (C,H,W)
            a = F.relu(out.detach()).to("cpu")
            feats[key] = a
        return hook

    # 注册钩子
    for k, m in name_to_module.items():
        if k in layer_names:
            handles.append(m.register_forward_hook(_mk_hook(k)))

    # 临时把 MaxPool2d 设 ceil_mode=True，避免 0×0
    for mod in model.modules():
        if isinstance(mod, nn.MaxPool2d):
            patched_pools.append((mod, mod.ceil_mode))
            mod.ceil_mode = True

    model.eval()
    with torch.no_grad():
        _ = model(x)

    # 还原 ceil_mode
    for mod, old in patched_pools:
        mod.ceil_mode = old

    for h in handles:
        h.remove()

    return feats

def score_channels(feat_chw: torch.Tensor, method: str = "energy"):
    """
    feat_chw: (C,H,W) 经过ReLU
    返回每个 channel 的分数 np.ndarray shape=(C,)
    method:
      - "energy": mean(a^2) 兼顾强度与稳定性（默认）
      - "variance": var(a) 空间变化强
      - "max": max(a) 极大响应
      - "sparse": 1 / (epsilon + 平均非零占比) ——偏爱稀疏、局部激活
    """
    a = feat_chw  # (C,H,W)
    C,H,W = a.shape
    x = a.view(C, -1)  # (C, H*W)
    if method == "energy":
        s = (x**2).mean(dim=1)
    elif method == "variance":
        s = x.var(dim=1, unbiased=False)
    elif method == "max":
        s = x.max(dim=1).values
    elif method == "sparse":
        nz = (x > 0).float().mean(dim=1)
        s = 1.0 / (1e-6 + nz)
    else:
        s = (x**2).mean(dim=1)
    return s.detach().cpu().numpy()

def pick_representative_channels(feat_chw: torch.Tensor, topk: int = 8, method: str = "energy", diversity: bool = True):
    """
    先按 method 打分取 3*topk，再做一次多样性筛选（避免挑到外观相近的通道）。
    返回: list[int] 长度<=topk
    """
    scores = score_channels(feat_chw, method=method)
    C = scores.shape[0]
    k0 = min(C, max(topk*3, topk))
    idx0 = np.argsort(-scores)[:k0]  # 初筛

    if not diversity or k0 <= topk:
        return idx0[:topk].tolist()

    # 简易多样性：用通道的 (H*W) 向量化后，贪心远离已选集合
    X = feat_chw[idx0].view(k0, -1).float().numpy()
    chosen = [idx0[0]]
    chosen_vecs = [X[0]]

    def _dist2(u, v):
        d = ((u - v)**2).mean()
        return d

    for i in range(1, k0):
        vi = X[i]
        dmin = min(_dist2(vi, vj) for vj in chosen_vecs)
        # 用 dmin 与原始分数做加权（兼顾可视冲击力与差异性）
        score_aug = scores[idx0[i]] + 0.25 * dmin
        # 暂存
        X[i, 0] = score_aug  # 偷放一下
    # 重新按“增强分数”排序
    order = np.argsort(-X[:,0])
    result = []
    used = set()
    for oi in order:
        cid = idx0[oi]
        if cid in used: 
            continue
        result.append(cid)
        used.add(cid)
        if len(result) >= topk:
            break
    return result


def plot_feature_grid(feat_chw: torch.Tensor, channels: list[int], ncols: int = 8, vmin=None, vmax=None, suptitle=None):
    """
    将 (C,H,W) 的若干通道画成单色热图网格
    """
    import math
    K = len(channels)
    ncols = min(ncols, K) if K>0 else 1
    nrows = int(math.ceil(K / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9*ncols, 1.9*nrows), dpi=160)
    if nrows*ncols == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    a = feat_chw.numpy()
    if vmin is None:
        vmin = float(a.min())
    if vmax is None:
        vmax = float(a.max())

    for i, ax in enumerate(axes.ravel()):
        if i < K:
            ch = channels[i]
            ax.imshow(a[ch], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"ch {ch}", fontsize=9)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(pad=0.3)
    return fig

# --- 对外：给定模型与样本，返回 {layer: (fig, meta)} ---
def build_layer_grids(model, x, topk_per_layer=8, method="energy"):
    """
    返回:
      grids: { 'conv1': (fig, {'picked': [..], 'C':C, 'H':H, 'W':W}) , ... }
    """
    feats = grab_conv_activations(model, x)
    grids = {}
    for lname, feat in feats.items():  # feat: (C,H,W)
        C,H,W = feat.shape
        picks = pick_representative_channels(feat, topk=topk_per_layer, method=method, diversity=True)
        fig = plot_feature_grid(
            feat, picks, ncols=min(8, topk_per_layer),
            # suptitle=f"{lname}: picked {len(picks)}/{C} channels"
        )
        grids[lname] = (fig, {"picked": picks, "C": C, "H": H, "W": W})
    return grids