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

def overlap_tensor2html(
    tensor: torch.Tensor,
    overlay: bool = True,
    highlight: list[tuple[int, int, int]] = []
) -> str:
    _, C, H, W = tensor.shape
    cell = 30
    gap = cell // 3
    shift_px = 0.6 * cell
    html_parts = ["<div class='tensor-vis'>"]

    for c in range(C):
        offset_top = offset_left = c * shift_px if overlay else 0

        # Build table rows
        rows_html = []
        for h in range(H):
            tds = []
            for w in range(W):
                val = arr[c, h, w]
                cell_str = "" if val is None else str(val)
                cls = "hl" if (c, h, w) in highlight else ""
                tds.append(f"<td class='{cls}'>{cell_str}</td>")
            rows_html.append("<tr>" + "".join(tds) + "</tr>")

        if overlay:
            style = f"position:absolute; top:{offset_top}px; left:{offset_left}px;"
        else:
            style = f"position:static; display:inline-block; margin-right:{gap}px;"

        table = (
            f"<table class='patch ch{c}' style='{style}'>"
            + "".join(rows_html) +
            "</table>"
        )
        html_parts.append(table)

    html_parts.append("</div>")
    return css + "\n".join(html_parts)