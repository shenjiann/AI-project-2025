import numpy as np
import torch
import torch.nn.functional as F

def generate_data(
    d_H: int = 3,
    d_W: int = 3,
    d_C: int = 1,
    f: int = 1,
    seed: int = 42
):
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
    return Z, W, Z0, dZ0, dZ, dW


def pad_matrix(
        matrix: torch.Tensor,
        pad: tuple[int, int, int, int] = (0, 0, 0, 0)):
    """
    对输入矩阵在左右上下添加 0 行/列
    """
    return F.pad(matrix, pad, mode='constant', value=0)

def matrix_to_html(
        matrix: torch.Tensor,
        highlight: list[tuple[int, int]] = None,
        prefix: str = None,
        wrap_math: bool = False # New parameter
    ) -> str:
    """
    将矩阵转换为带左右中括号的 HTML 表格，并高亮特定元素。
    支持形状为 [1, 1, H, W] 的 PyTorch Tensor
    """
    if highlight is None:
        highlight = []

    # 如果是 PyTorch tensor，转换为 NumPy 并 squeeze 成 2D 或 3D
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    if matrix.ndim == 4 and matrix.shape[0] == 1 and matrix.shape[1] == 1:
        matrix = matrix[0, 0]  # 变为 (H, W)


    matrix_html = '<div class="matrix-container"><table class="matrix">'
    for i in range(matrix.shape[0]):
        matrix_html += '<tr>'
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if (i, j) in highlight:
                matrix_html += f'<td class="highlight">{val}</td>'
            else:
                matrix_html += f'<td>{val}</td>'
        matrix_html += '</tr>'
    matrix_html += '</table></div>'

    return matrix_html
