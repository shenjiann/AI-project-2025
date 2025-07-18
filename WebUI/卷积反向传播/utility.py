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
    shift_px = 0.6*cell if overlay else 0

    container_style = ""
    if overlay:
        total_shift      = shift_px * (C - 1)
        container_width  = W * cell + total_shift
        container_height = H * cell + total_shift
        container_style  = (
            f"style='width:{container_width}px; "
            f"height:{container_height}px;'"
        )

    html_parts = [f"<div class='tensor-vis' {container_style}>"]


    arr = tensor.detach().cpu().numpy()[0]

    for c in range(C):
        if overlay:
            # channel 0 放在最左下，其余向右上偏移
            offset_top  = (C - 1 - c) * shift_px   # 越往后的 channel 越靠上
            offset_left = c * shift_px             # 越往后的 channel 越靠右
        else:
            offset_top  = 0
            offset_left = 0

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
            z_index = C - c
            style = (
                f"position:absolute; top:{offset_top}px; left:{offset_left}px;"
                f" z-index:{z_index};"
            )
        else:
            gap = cell // 3  # spacing when overlay=False
            style = f"display:inline-block; margin-right:{gap}px;"

        html_parts.append(
            f"<table class='patch ch{c}' style='{style}'>"
            + "".join(rows_html) +
            "</table>"
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)
