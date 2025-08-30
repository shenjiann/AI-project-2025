# deconv_visualize.py
import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from AlexNet import AlexNet


# ============================
# 反池化层封装
# ============================
class Unpool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(Unpool, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x, indices):
        return nn.functional.max_unpool2d(
            x, indices, self.kernel_size, self.stride, self.padding
        )


# ============================
# 工具：收集前向每层输出与池化索引（按层号绑定）
# ============================
@torch.no_grad()
def forward_collect(model: nn.Module, image: torch.Tensor):
    """
    返回：
      - feature_by_layer: {layer_index: tensor_after_this_layer}
      - pool_indices_by_layer: {layer_index: indices_of_this_pool}
      - input_after_layer[-1]: image （便于统一处理）
    """
    feature_by_layer = {-1: image}  # 把输入也放进来，作为“第 -1 层”的输出
    pool_indices_by_layer = {}
    x = image
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.MaxPool2d):
            # 需要 layer.return_indices=True 才能返回 (x, indices)
            x, indices = layer(x)
            pool_indices_by_layer[i] = indices
        else:
            x = layer(x)
        feature_by_layer[i] = x
    return feature_by_layer, pool_indices_by_layer


def conv_layer_indices(model: nn.Module):
    """自动找出所有 Conv2d 的层号（在 model.features 里的索引）"""
    idxs = []
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            idxs.append(i)
    return idxs


# ============================
# 构建镜像反向网络（并拷贝权重）
# 记录每个反向层的“绑定信息”：
#   - Unpool: ('unpool', forward_layer_index)
#   - Deconv: ('deconv', forward_conv_index)
# ============================
def build_deconv_from(model: nn.Module, upto_layer_i: int):
    deconv_layers = []
    layer_tags = []  # 与 deconv_layers 同长；记录 ('unpool'| 'deconv', forward_idx)

    for i in range(upto_layer_i, -1, -1):
        f = model.features[i]
        if isinstance(f, nn.ReLU):
            deconv_layers.append(nn.ReLU())
            layer_tags.append(None)
        elif isinstance(f, nn.LocalResponseNorm):
            # 常规做法：忽略 LRN
            pass
        elif isinstance(f, nn.MaxPool2d):
            u = Unpool(f.kernel_size, f.stride, f.padding)
            deconv_layers.append(u)
            layer_tags.append(('unpool', i))  # 绑定这个 Unpool 对应的前向层号
        elif isinstance(f, nn.Conv2d):
            # 镜像卷积：ConvTranspose2d；output_padding 将在运行时动态设置
            deconv = nn.ConvTranspose2d(
                in_channels=f.out_channels,
                out_channels=f.in_channels,
                kernel_size=f.kernel_size,
                stride=f.stride,
                padding=f.padding,
                output_padding=0,  # 先放 0，运行时动态修正
                bias=(f.bias is not None),
            )
            with torch.no_grad():
                deconv.weight.data = f.weight.data.clone()
                if deconv.bias is not None:
                    deconv.bias.data.zero_()

            deconv_layers.append(deconv)
            layer_tags.append(('deconv', i))

    return nn.Sequential(*deconv_layers), layer_tags


def _to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def _compute_needed_output_padding(in_hw, target_hw, stride, padding, kernel_size):
    """
    对 ConvTranspose2d 的每一维，计算所需 output_padding：
      out = (in - 1)*s - 2p + k + output_padding
    => output_padding = target - ((in - 1)*s - 2p + k)
    结果裁剪到 [0, s-1]，并返回 2-tuple
    """
    in_h, in_w = in_hw
    tgt_h, tgt_w = target_hw
    s_h, s_w = _to_2tuple(stride)
    p_h, p_w = _to_2tuple(padding)
    k_h, k_w = _to_2tuple(kernel_size)

    pred_h = (in_h - 1) * s_h - 2 * p_h + k_h
    pred_w = (in_w - 1) * s_w - 2 * p_w + k_w

    op_h = tgt_h - pred_h
    op_w = tgt_w - pred_w

    # 合法范围：[0, stride-1]
    op_h = max(0, min(op_h, s_h - 1))
    op_w = max(0, min(op_w, s_w - 1))
    return (op_h, op_w)


@torch.no_grad()
def run_deconv(
    deconv_net: nn.Sequential,
    layer_tags: list,
    pool_indices_by_layer: dict,
    feature_by_layer: dict,
    start_feat: torch.Tensor,
    input_hw: tuple,
):
    """
    执行反向网络。
    - Unpool：用绑定的前向层号找 indices。
    - Deconv：动态计算 output_padding，使输出精确匹配“该卷积层的前一层输出尺寸”（第一层卷积匹配输入尺寸）。
    """
    x = start_feat
    for layer, tag in zip(deconv_net, layer_tags):
        if isinstance(layer, Unpool):
            assert tag and tag[0] == 'unpool'
            f_idx = tag[1]
            assert f_idx in pool_indices_by_layer, (
                f"Missing pool indices for forward layer {f_idx}. "
                "请检查 MaxPool2d 是否 return_indices=True。"
            )
            x = layer(x, pool_indices_by_layer[f_idx])
        elif isinstance(layer, nn.ConvTranspose2d):
            assert tag and tag[0] == 'deconv'
            f_idx = tag[1]
            # 目标尺寸：该卷积层前一层输出的空间尺寸；如果 f_idx == 0，则目标就是输入图像尺寸
            if (f_idx - 1) in feature_by_layer:
                target = feature_by_layer[f_idx - 1].shape[-2:]  # (H, W)
            else:
                # 保险兜底；理论上不会走到
                target = input_hw

            in_hw = x.shape[-2:]  # 当前特征尺寸
            s = layer.stride
            p = layer.padding
            k = layer.kernel_size

            op_h, op_w = _compute_needed_output_padding(in_hw, target, s, p, k)
            # 动态设置 output_padding
            layer.output_padding = (op_h, op_w)
            x = layer(x)
        else:
            x = layer(x)
    return x


# ============================
# 选择最大激活：整通道/单个神经元 两种模式
# ============================
def pick_activation(target_feature_map: torch.Tensor, mode: str = "channel"):
    """
    target_feature_map: [1, C, H, W]
    mode: "channel" 或 "single"
    """
    assert target_feature_map.ndim == 4 and target_feature_map.shape[0] == 1
    fmap = target_feature_map[0]  # [C,H,W]
    C, H, W = fmap.shape
    flat_idx = torch.argmax(fmap).item()
    c = flat_idx // (H * W)

    selected = torch.zeros_like(target_feature_map)
    if mode == "channel":
        selected[0, c, :, :] = target_feature_map[0, c, :, :]
    else:  # "single"
        h = (flat_idx % (H * W)) // W
        w = (flat_idx % (H * W)) % W
        selected[0, c, h, w] = target_feature_map[0, c, h, w]
    return selected


# ============================
# 顶层可视化函数（保证最终恢复到 input_hw）
# ============================
def visualize_features(model, image, layer_to_visualize: int, mode: str = "channel"):
    """
    可视化第 layer_to_visualize 个卷积层的激活投影回像素空间。
    mode: "channel"（整通道）或 "single"（单神经元）
    返回：H×W×3 的 numpy 数组（0~1），H/W == 输入图像尺寸（例如 227×227）
    """
    model.eval()

    # 前向：收集每层输出与池化索引（按层号绑定）
    feature_by_layer, pool_indices_by_layer = forward_collect(model, image)

    # 自动获取卷积层索引
    conv_idxs = conv_layer_indices(model)
    if not (0 <= layer_to_visualize < len(conv_idxs)):
        print(
            f"[Warn] layer_to_visualize={layer_to_visualize} 越界；共有 {len(conv_idxs)} 个卷积层。"
        )
        return None

    target_i = conv_idxs[layer_to_visualize]
    target_feature_map = feature_by_layer[target_i]  # [1, C, H, W]

    if target_feature_map.ndim != 4 or target_feature_map.shape[0] != 1:
        print(
            f"[Warn] 目标层输出形状异常：{tuple(target_feature_map.shape)}，跳过可视化。"
        )
        return None

    # 选最大激活（整通道或单点）
    selected_feature_map = pick_activation(
        target_feature_map, mode=mode
    ).to(image.device)

    # 构建反向网络（从 target_i 倒序到 0）
    deconv_net, layer_tags = build_deconv_from(model, target_i)

    # 运行反向（动态设置每层 ConvTranspose2d 的 output_padding）
    input_hw = tuple(image.shape[-2:])  # (H, W)，例如 (227, 227)
    with torch.no_grad():
        current = run_deconv(
            deconv_net,
            layer_tags,
            pool_indices_by_layer,
            feature_by_layer,
            selected_feature_map,
            input_hw=input_hw,
        )

    # 数值安全归一化 + 转置到 [H,W,3]
    arr = current.squeeze(0).detach().cpu().numpy()  # [3,H,W]（期望）
    eps = 1e-8
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - mn) / (mx - mn)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# ============================
# 从本地 www/cifar10_samples 中随机挑选一张图片
# 目录结构：
#   BASE_DIR/www/cifar10_samples/<class>/*.png|jpg|jpeg|bmp|webp
# ============================
def pick_random_local_image(samples_root: str, seed: int = None):
    if seed is not None:
        random.seed(seed)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    all_paths = []
    # 避免 samples_root 不存在时报错
    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"{samples_root} not found.")
    class_dirs = sorted([d for d in next(os.walk(samples_root))[1]])
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirs under {samples_root}")

    for cls_dir in class_dirs:
        for ext in exts:
            all_paths.extend(glob.glob(os.path.join(samples_root, cls_dir, ext)))

    if not all_paths:
        raise FileNotFoundError(
            f"No images found under {samples_root}/* with extensions {exts}"
        )

    img_path = random.choice(all_paths)
    class_name = os.path.basename(os.path.dirname(img_path))
    return img_path, class_name


# ============================
# 主程序
# ============================
if __name__ == "__main__":
    # === 可选：固定随机种子，保证每次选择同一张样本图像 ===
    FIX_SEED = True
    SEED = 420
    if FIX_SEED:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    # 路径配置
    BASE_DIR = "./ShinyApps/卷积层输出可视化"
    SAMPLES_DIR = os.path.join(BASE_DIR, "www", "cifar10_samples")
    OUTPUT_DIR = BASE_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 预处理（AlexNet 常用 227×227 + ImageNet 规范化）
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # 反归一化（仅用于把输入图保存回可视范围）
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    # 随机挑选一张本地样本图
    img_path, class_name = pick_random_local_image(SAMPLES_DIR, 43)
    print(f"Picked sample: {img_path} (class: {class_name})")

    # 读取并变换
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img)  # [3,227,227]

    # 模型
    model = AlexNet().to(device)
    weight_path = os.path.join(BASE_DIR, "models", "AlexNet.pth")
    try:
        sd = torch.load(weight_path, map_location=device)
        model.load_state_dict(sd)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {weight_path} not found. Please ensure it's in the correct path.")
        raise SystemExit(1)

    # 让 MaxPool2d 返回 indices（猴补丁）
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.MaxPool2d) and not layer.return_indices:
            model.features[i] = nn.MaxPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                return_indices=True,  # 关键！
                ceil_mode=layer.ceil_mode,
            )

    model.eval()

    # 保存原始输入图片（反归一化）
    with torch.no_grad():
        inv_img = inv_normalize(img_tensor.clone()).clamp(0.0, 1.0).numpy()
        inv_img = np.transpose(inv_img, (1, 2, 0))
    plt.imshow(inv_img)
    plt.title(f"Original Input Image\n{os.path.basename(img_path)}  |  class={class_name}")
    plt.axis("off")
    input_out = os.path.join(OUTPUT_DIR, "input_image.png")
    plt.savefig(input_out, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved: {input_out}")

    # 扩展 batch 维度
    img_batch = img_tensor.unsqueeze(0).to(device)  # [1,3,227,227]

    # 可视化每个卷积层
    conv_idxs = conv_layer_indices(model)
    print(f"Conv layer indices in model.features: {conv_idxs}")
    n_convs = len(conv_idxs)

    # 可选：channel 或 single
    VIS_MODE = "channel"  # or "single"

    for k in range(n_convs):
        print(f"Visualizing for conv layer #{k+1}/{n_convs} (features idx={conv_idxs[k]}) ...")
        vis = visualize_features(model, img_batch, k, mode=VIS_MODE)
        if vis is None:
            print(f"[Skip] Layer {k+1} visualization returned None.")
            continue
        # 验证尺寸是否为 227×227
        assert vis.shape[0] == 227 and vis.shape[1] == 227, f"Unexpected size: {vis.shape}"
        plt.imshow(vis)
        plt.title(f"Layer {k+1} Visualization ({VIS_MODE})\n{os.path.basename(img_path)}  |  class={class_name}")
        plt.axis("off")
        out_path = os.path.join(OUTPUT_DIR, f"layer_{k+1}_visualization.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

    print("All visualizations saved as PNG files.")