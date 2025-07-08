import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

def overlap_tensor2html(
    tensor: torch.Tensor,
    overlay: bool = True,
    highlight: list[tuple[int, int, int]] = [],
    cell = 30,
) -> str:
    _, C, H, W = tensor.shape
    gap = cell // 3
    shift_px = 0.6*cell if overlay else 0

    html_parts = [
        f"<div class='tensor-vis'>"
    ]
    arr = tensor.detach().cpu().numpy()[0]

    for c in range(C):
        offset_top = offset_left = c * shift_px if overlay else 0

        # Build table rows
        rows_html = []
        for h in range(H):
            tds = []
            for w in range(W):
                val = arr[c, h, w]
                if val is None:
                    cell_str = ""
                else:
                    cell_str = str(int(val)) if val == int(val) else f'{val:2g}'
                cls = "hl" if (c, h, w) in highlight else ""
                tds.append(f"<td class='{cls}'>{cell_str}</td>")
            rows_html.append("<tr>" + "".join(tds) + "</tr>")

        if overlay:
            style = f"position:absolute; top:{offset_top}px; left:{offset_left}px;"
        else:
            style = f"display:inline-block; margin-right:{gap}px;"

        html_parts.append(
            f"<table class='patch ch{c}' style='{style}'>"
            + "".join(rows_html) +
            "</table>"
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)
