from shiny import App, ui, render
import numpy as np

###############################################################################
# 一些演示用的假数据 —— 三个 2×5 的 patch，及其相对位移（dy, dx，单位：格子，可为非整数）
###############################################################################
cell_px = 40  # 每个单元格的像素宽/高
patches = [
    dict(
        matrix=np.arange(10).reshape(2, 5),
        offset=(0, 0),        # 基准 patch
        css_class="orange",
    ),
    dict(
        matrix=np.array([[10, 11, 12, 13, 14],
                         [15, 16, 17, 18, 19]]),
        offset=(0.6, 0.6),        # 向下、向右各移 1.2, 0.6 格
        css_class="blue",
    ),
]

# ----- demo tensor for UI preview -----
tensor_demo = np.stack([p["matrix"] for p in patches])  # shape (C,H,W)
# --------------------------------------

###############################################################################
# 把上面数据转成一段完整的 HTML + 内联 CSS
###############################################################################
def overlapping_patches_to_html(
    tensor: np.ndarray,
    cell: int = 40,
    overlay: bool = True,
    highlight: list[tuple[int, int, int]] | None = None,
) -> str:
    """
    Render a (C, H, W) or (N, C, H, W) tensor into HTML tables.

    Parameters
    ----------
    tensor
        NumPy array with shape (C, H, W) **or** (N, C, H, W).  The leading
        batch/patch dimension *N* is ignored (assumed 0/1).  `C` is the channel
        count (1 or 2).
    cell
        Pixel size of each table cell (both width & height).
    overlay
        If True, channels are drawn on the same coordinate system with a small
        offset (0 px for ch‑0, 0.6 × cell for ch‑1 …).  If False, channels are
        placed side‑by‑side (`inline‑block`).
    highlight
        List of tuples ``(c, h, w)``.  Any cell whose **channel**, **row
        (height index)**, and **column (width index)** matches will be colored
        orange.

    Returns
    -------
    str
        A chunk of HTML (including inline CSS) ready for `ui.HTML(...)`.
    """
    # Normalize shape → (C, H, W)
    arr = np.asarray(tensor)
    if arr.ndim == 4:       # (N, C, H, W) → use the first sample
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("tensor must have shape (C,H,W) or (N,C,H,W)")
    C, H, W = arr.shape

    highlight = set(highlight or [])

    # --- CSS -----------------------------------------------------------------
    css = f"""
    <style>
      .tensor-vis  {{ position:relative; }}
      .patch       {{ border-collapse:collapse; }}
      .patch td    {{ width:{cell}px; height:{cell}px;
                      border:1px solid #000;
                      text-align:center; vertical-align:middle; }}
      /* Channel colours */
      .ch0 td {{ background:rgb(220,240,255); }}  /* 淡蓝 */
      .ch1 td {{ background:rgb(220,255,220); }}  /* 淡绿 */
      /* Highlight */
      .hl  {{ background:orange !important; }}
    </style>
    """

    gap = cell // 3  # spacing when overlay=False
    html_parts = ["<div class='tensor-vis'>"]

    # Small shift for overlay view
    shift_px = 0.6 * cell

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

###############################################################################
# Shiny UI & Server
###############################################################################
app_ui = ui.page_fluid(
    ui.input_checkbox("overlay", "重叠显示", True),
    ui.output_ui("tensor_vis"),
)

def server(input, output, session):
    @render.ui
    def tensor_vis():
        return ui.HTML(
            overlapping_patches_to_html(
                tensor_demo,
                cell_px,
                overlay=input.overlay(),
                highlight=[]
            )
        )

app = App(app_ui, server)