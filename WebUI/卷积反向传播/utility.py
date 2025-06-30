import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
Z = torch.randint(
    low=-5, high=6, size=(1, 1, 3, 3), 
    dtype=torch.float32, requires_grad=True)
W = torch.randint(
    low=-3, high=4, size=(1, 1, 2, 2), 
    dtype=torch.float32, requires_grad=True)
Z0 = F.conv2d(Z, W)
dZ0 = torch.randint(-3, 3, Z0.shape, dtype=torch.float32)
Z0.backward(dZ0)
dZ = Z.grad
dW = W.grad

def matrix_to_html(matrix: np.ndarray | torch.Tensor, highlight: list[tuple[int, int]] = None) -> str:
    """
    将矩阵转换为带左右中括号的 HTML 表格，并高亮特定元素。
    支持形状为 [1, 1, H, W] 的 PyTorch Tensor。
    """
    import torch  # 防止类型未定义
    if highlight is None:
        highlight = []

    # 如果是 PyTorch tensor，转换为 NumPy 并 squeeze 成 2D
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    if matrix.ndim == 4 and matrix.shape[0] == 1 and matrix.shape[1] == 1:
        matrix = matrix[0, 0]  # 变为 (H, W)

    html = '<div class="matrix-container"><table class="matrix">'
    for i in range(matrix.shape[0]):
        html += '<tr>'
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if (i, j) in highlight:
                html += f'<td class="highlight">{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</table></div>'
    return html