import numpy as np
import torch
import torch.nn.functional as F

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
        'dW^{[l]}': dW
    }

def pad_matrix(
        matrix: torch.Tensor,
        pad: tuple[int, int, int, int] = (0, 0, 0, 0)):
    """
    对输入矩阵在左右上下添加 0 行/列
    """
    return F.pad(matrix, pad, mode='constant', value=0)

def tensor2html(
        tensor: torch.Tensor,
        highlight: list[tuple[int, int, int]] = None,
    ) -> list[str]:
    if highlight is None:
        highlight = []

    def _matrix_to_html_2d(mat: np.ndarray) -> str:
        html = '<div class="matrix-container"><table class="matrix">'
        for i in range(mat.shape[0]):
            html += '<tr>'
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if (i, j) in highlight:
                    html += f'<td class="highlight">{val}</td>'
                else:
                    html += f'<td>{val}</td>'
            html += '</tr>'
        html += '</table></div>'
        return html
    
    html_list = []
    if tensor.shape[0] == 1 and (tensor.shape[1] == 1 or tensor.shape[1] == 2):
        for c in range(tensor.shape[1]):
            html_list.append(_matrix_to_html_2d(tensor[0, c]))
    else:
        raise ValueError("Only support input of shape [1, C, H, W] where C=1 or C=2")
    
    return html_list

