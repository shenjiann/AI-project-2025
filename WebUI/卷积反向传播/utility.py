import numpy as np
import torch

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