import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

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

