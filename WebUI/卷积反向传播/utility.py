import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

def generate_data(
    d_H: int = 3,
    d_W: int = 3,
    d_C: int = 1,
    f: int = 1,
    seed: int = 42
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    Z = torch.randint(
        low=-5, high=6, size=(1, d_C, d_H, d_W), 
        dtype=torch.float32, requires_grad=True
    )
    W = torch.randint(
        low=-3, high=4, size=(1, d_C, f, f), 
        dtype=torch.float32, requires_grad=True
    )
    Z0 = F.conv2d(Z, W)
    dZ0 = torch.randint(
        low=-3, high=3, size=Z0.shape, 
        dtype=torch.float32
    )
    Z0.backward(dZ0)
    dZ = Z.grad
    dW = W.grad
    return {
        'Z^{[l-1]}': Z,
        'W^{[l]}': W,
        'Z_0^{[l]}': Z0,
        'dZ_0^{[l]}': dZ0,
        'dZ^{[l-1]}': dZ,
        'dW^{[l]}': dW,
        'Z0shape': Z0.shape,
    }

# def pad_matrix(
#         matrix: torch.Tensor,
#         pad: tuple[int, int, int, int] = (0, 0, 0, 0)):
#     """
#     对输入矩阵在左右上下添加 0 行/列
#     """
#     return F.pad(matrix, pad, mode='constant', value=0)

def tensor2html(
        tensor: torch.Tensor,
        highlight: list[tuple[int, int, int]] = None,
    ) -> list[str]:
    """
    将tensor转为html, 返回list长度为tensor的channel数
    """
    if highlight is None:
        highlight = []
    
    def _matrix_to_html_2d(mat: np.ndarray,
                           hl_2d: List[Tuple[int, int]]) -> str:
        html = '<div class="matrix-container"><table class="matrix">'
        for i in range(mat.shape[0]):
            html += "<tr>"
            for j in range(mat.shape[1]):
                val = mat[i, j]
                cell_cls = "highlight" if (i, j) in hl_2d else ""
                html += f'<td class="{cell_cls}">{val}</td>'
            html += "</tr>"
        html += "</table></div>"
        return html

    html_list = []
    C = tensor.shape[1]

    for c in range(C):
        hl_2d = [(i, j) for (chan, i, j) in highlight if chan == c]
        html_list.append(_matrix_to_html_2d(tensor[0, c].detach().cpu().numpy(), hl_2d))

    return html_list


def get_focus_coords(
    tensor_shape: Tuple[int, int, int, int],
    steps: int
) -> List[Tuple[int, int, int]]:
    """
    根据 step 计算 (channel, row, col) 坐标，用于在可视化中高亮张量元素。

    参数
    ----
    tensor_shape : (1, C, H, W)
        张量的形状，只支持 batch_size = 1，通道 C 取 1 或 2。
    step : int
        第几个“关注”步骤。
        * step = 0  ⇒ 返回空列表 []（无需高亮）
        * 其他正整数 ⇒ 返回一个单元素列表 [(c, i, j)]
          - 先按 channel 维扫描，再在每个 channel 内按行优先顺序扫描。

    返回
    ----
    List[(c, i, j)]
        需要高亮的元素坐标列表。若 step 超出范围则返回 []。
    """
    # 初始返回空坐标列表
    if steps == 0:
        return []

    # 解包形状并做基本校验
    _, C, H, W = tensor_shape
    if C not in (1, 2):
        raise ValueError("仅支持通道数 C 为 1 或 2 的张量")

    elems_per_channel = H * W
    total_elems = C * elems_per_channel

    # 越界，返回空坐标列表
    if steps > total_elems:
        return []
    
    # 将 step-1 映射到 (channel, row, col)
    idx = steps - 1                     # 转为 0-based
    c = idx // elems_per_channel       # 所在通道
    in_channel_idx = idx % elems_per_channel
    i = in_channel_idx // W            # 行
    j = in_channel_idx % W             # 列

    return [(c, i, j)]