# alexnet_cifar10_vis.py
# 可复现 Zeiler & Fergus 图2风格的可视化：AlexNet + CIFAR-10
# 依赖: torch torchvision pillow numpy
# 运行: python alexnet_cifar10_vis.py  (修改参数区即可)

import os, heapq, numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils

# =========================
# ======== 参数区 =========
# =========================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH      = "layer5_vis.png"   # 输出拼板图片
LAYER_IDX     = 12                 # AlexNet.features 的层索引：6/8≈中层，10≈layer4，12≈layer5
N_MAPS        = 8                  # 随机选择多少个通道可视化
TOPK          = 9                  # 每个通道选多少个最强激活（论文取9）
DATA_LIMIT    = None               # 仅用前N张样本做检索（调试时设小一点）
BATCH_SIZE    = 256
NUM_WORKERS   = 4
SEED          = 42

# 是否在 CIFAR-10 上做一次快速微调（只为可能的好看一点；非必须）
DO_FINETUNE   = False
FINETUNE_EPOCHS = 5
FINETUNE_LR     = 1e-3
FREEZE_FEATURES  = False  # True=只训分类头；False=特征也解冻

# =========================
# ======== 工具函数 ========
# =========================
def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s)

def make_cifar10_loader(split="train", limit=DATA_LIMIT):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = datasets.CIFAR10(root="./data", train=(split=="train"),
                          transform=tfm, download=True)
    if limit is not None:
        ds.data = ds.data[:limit]
        ds.targets = ds.targets[:limit]
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

class MaxPool2dRecord(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding,
                                 ceil_mode=ceil_mode, return_indices=True)
        self.last_indices = None
        self.last_input_shape = None

    def forward(self, x):
        y, idx = self.pool(x)
        self.last_indices = idx
        self.last_input_shape = x.shape
        return y

def alexnet_with_indices(local_weights_path: str | None = None):
    m = models.alexnet(weights=None)
    if local_weights_path is not None:
        state = torch.load(local_weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state, strict=True)
    else:
        try:
            m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        except Exception as e:
            print("[WARN] 预训练权重加载失败，使用随机初始化：", e)

    feats = []
    for layer in m.features:
        if isinstance(layer, nn.MaxPool2d):
            feats.append(MaxPool2dRecord(kernel_size=layer.kernel_size,
                                         stride=layer.stride,
                                         padding=layer.padding,
                                         ceil_mode=layer.ceil_mode))
        else:
            feats.append(layer)
    m.features = nn.Sequential(*feats)
    return m

class FeatureCacher:
    def __init__(self, features: nn.Sequential):
        self.features = features
        self.fmaps, self.pool_indices, self.pool_in_shapes = {}, {}, {}
        self.handles = []
        for i, layer in enumerate(self.features):
            if isinstance(layer, MaxPool2dRecord):
                self.handles.append(layer.register_forward_hook(self._pool_hook(i)))
            else:
                self.handles.append(layer.register_forward_hook(self._feat_hook(i)))

    def _feat_hook(self, idx):
        def fn(module, inp, out):
            self.fmaps[idx] = out.detach()
        return fn

    def _pool_hook(self, idx):
        def fn(module: MaxPool2dRecord, inp, out):
            # out 是 y（Tensor），indices 存在 module.last_indices
            self.fmaps[idx] = out.detach()
            self.pool_indices[idx] = module.last_indices.detach()
            self.pool_in_shapes[idx] = module.last_input_shape  # pool 前尺寸
        return fn

    def close(self):
        for h in self.handles: h.remove()

class DeconvNet(nn.Module):
    """ConvTranspose + MaxUnpool + ReLU，对偶AlexNet.features"""
    def __init__(self, features: nn.Sequential):
        super().__init__()
        self.features = features
        self.de_layers = nn.ModuleList()
        for layer in reversed(self.features):
            if isinstance(layer, nn.Conv2d):
                de = nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                )
                de.weight.data = layer.weight.data
                if layer.bias is not None: de.bias.data.zero_()
                self.de_layers.append(de)
            elif isinstance(layer, nn.ReLU):
                self.de_layers.append(nn.ReLU(inplace=False))
            elif isinstance(layer, nn.MaxPool2d):
                self.de_layers.append(nn.MaxUnpool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                ))
            else:
                self.de_layers.append(nn.Identity())

    def forward(self, caches: FeatureCacher, start_layer_idx: int, sparse_act: torch.Tensor):
        x = sparse_act
        L = len(self.features)
        for i in range(start_layer_idx, -1, -1):
            layer = self.features[i]
            de_layer = self.de_layers[L-1-i]
            if isinstance(layer, nn.Conv2d):
                x = de_layer(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                indices = caches.pool_indices[i]
                out_size = caches.pool_in_shapes[i]
                x = de_layer(x, indices, output_size=out_size)
            else:
                pass
        return x

def inv_normalize(x):
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    return torch.clamp(x*std + mean, 0, 1)

@torch.no_grad()
def topk_activations(model, loader, layer_idx, channel, K=TOPK):
    feats = model.features
    cacher = FeatureCacher(feats)
    topk = []  # 存 (score: float, tie_breaker: int, payload)
    tie_breaker = 0

    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        _ = feats(imgs)
        fmap = cacher.fmaps[layer_idx]                   # [B,C,H,W]
        # vals/idxs: [B, 1]  ->  [B]
        vals, idxs = torch.topk(fmap[:, channel, :, :].flatten(1), k=1)
        vals = vals.squeeze(1)                           # [B]
        idxs = idxs.squeeze(1)                           # [B]
        B, H, W = fmap.shape[0], fmap.shape[2], fmap.shape[3]

        for b in range(B):
            score = float(vals[b].item())                # ★ Python float
            flat  = int(idxs[b].item())                  # ★ Python int
            ph, pw = divmod(flat, W)                     # 两个 Python int
            img_cpu = imgs[b].detach().cpu()             # 这是 Tensor，但放到 payload 里

            payload = (img_cpu, ph, pw, (H, W))          # 不参与比较
            if len(topk) < K:
                topk.append((score, tie_breaker, payload))
                topk.sort(key=lambda t: t[0])            # 小根序（可用 heapq，也可直接排序）
            else:
                # 维护小根堆逻辑：如果更大，就替换最小的
                if score > topk[0][0]:
                    topk[0] = (score, tie_breaker, payload)
                    topk.sort(key=lambda t: t[0])
            tie_breaker += 1

    cacher.close()
    # 从大到小输出 payload
    topk.sort(key=lambda t: -t[0])
    return [t[2] for t in topk]

@torch.no_grad()
def reconstruct_one(model, deconv, layer_idx, channel, img_tensor, pos_hw, fmap_hw):
    feats = model.features
    cacher = FeatureCacher(feats)
    x = img_tensor.unsqueeze(0).to(DEVICE)
    _ = feats(x)

    fmap = cacher.fmaps[layer_idx].clone()
    fmap.zero_()
    ph, pw = pos_hw
    fmap[0, channel, ph, pw] = 1.0

    recon = deconv(cacher, layer_idx, fmap)[0]
    cacher.close()

    recon = inv_normalize(torch.clamp(recon.cpu(), -2.5, 2.5))
    # 近似感受野裁剪
    Hf, Wf = fmap_hw
    cy = int((ph + 0.5) * 224 / Hf)
    cx = int((pw + 0.5) * 224 / Wf)
    half = 28
    y0, y1 = max(0, cy-half), min(224, cy+half)
    x0, x1 = max(0, cx-half), min(224, cx+half)
    x_disp = inv_normalize(img_tensor.cpu())
    patch = x_disp[:, y0:y1, x0:x1]
    patch = F.interpolate(patch.unsqueeze(0), size=(112,112), mode="bilinear", align_corners=False)[0]
    recon_small = F.interpolate(recon.unsqueeze(0), size=(112,112), mode="bilinear", align_corners=False)[0]
    return patch, recon_small

def visualize_layer(model, loader, layer_idx=LAYER_IDX, n_maps=N_MAPS, out_path=OUT_PATH):
    set_seed()
    model.eval()
    feats = model.features

    # 先获取该层通道数
    cacher = FeatureCacher(feats)
    imgs,_ = next(iter(loader))
    _ = feats(imgs[:1].to(DEVICE))
    C = cacher.fmaps[layer_idx].shape[1]
    cacher.close()

    channels = np.random.choice(C, size=min(n_maps, C), replace=False)
    deconv = DeconvNet(feats).to(DEVICE).eval()

    rows = []
    for ch in channels:
        topk = topk_activations(model, loader, layer_idx, ch, K=TOPK)
        tiles_patch, tiles_recon = [], []
        for score, img_t, ph, pw, fmap_hw in topk:
            p, r = reconstruct_one(model, deconv, layer_idx, ch, img_t, (ph,pw), fmap_hw)
            tiles_patch.append(p); tiles_recon.append(r)
        col_left  = utils.make_grid(torch.stack(tiles_recon,0), nrow=3, padding=2)
        col_right = utils.make_grid(torch.stack(tiles_patch,0), nrow=3, padding=2)
        rows.append(torch.cat([col_left, col_right], dim=2))
    panel = torch.cat(rows, dim=1)
    utils.save_image(panel, out_path)
    print(f"[Saved] {out_path}")

def finetune_on_cifar10(model, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, freeze_features=FREEZE_FEATURES):
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 10)
    if freeze_features:
        for p in model.features.parameters(): p.requires_grad_(False)
    model.to(DEVICE).train()

    train_loader = make_cifar10_loader("train")
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for ep in range(1, epochs+1):
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optim.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward(); optim.step()
            running += loss.item()*imgs.size(0)
        print(f"[finetune] epoch {ep}/{epochs} loss={running/len(train_loader.dataset):.4f}")
    model.eval()
    return model

# =========================
# ========= 主程 ==========
# =========================
def main():
    set_seed()
    loader = make_cifar10_loader("train", limit=DATA_LIMIT)
    model = alexnet_with_indices(local_weights_path="Shiny应用/CIFAR/models/alexnet-owt-7be5be79.pth")
    if DO_FINETUNE:
        model = finetune_on_cifar10(model)
    else:
        model = model.to(DEVICE).eval()

    visualize_layer(model, loader, layer_idx=LAYER_IDX, n_maps=N_MAPS, out_path=OUT_PATH)

if __name__ == "__main__":
    main()