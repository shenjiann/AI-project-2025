# vector_viz.py
from __future__ import annotations
import numpy as np
from matplotlib import cm, colors as mpl_colors
from html import escape

def _to_hex_color(x: float, vmin: float, vmax: float, cmap_name: str = "coolwarm") -> str:
    # 归一化并用 matplotlib colormap 映射为 hex
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)
    r, g, b, _ = cmap(norm(x))
    return mpl_colors.to_hex((r, g, b), keep_alpha=False)

def render_vector_html(vec: np.ndarray,
                       cmap: str = "coolwarm",
                       vmin: float | None = None,
                       vmax: float | None = None,
                       dot_px: int = 28,
                       gap_px: int = 10,
                       show_index: bool = True) -> str:
    """
    把一维向量渲染为水平可滚动的一排圆点（颜色表示数值）。
    返回 HTML 字符串，可直接丢给 Shiny 的 ui.HTML().
    """
    vec = np.asarray(vec).ravel()
    if vec.size == 0:
        return "<div>（向量为空）</div>"

    if vmin is None: vmin = float(np.nanmin(vec))
    if vmax is None: vmax = float(np.nanmax(vec))
    # 若全相等，避免除零
    if vmin == vmax:
        vmin = vmax - 1.0

    # 容器样式：横向滚动 + 居中对齐
    container_css = (
        "overflow-x: auto; white-space: nowrap; padding: 8px 4px;"
        "border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa;"
    )
    # 单个圆点样式（大小、圆角、阴影、居中）
    dot_size = f"width:{dot_px}px; height:{dot_px}px;"
    dot_base_css = (
        f"display:inline-flex; {dot_size} border-radius:50%; "
        "align-items:center; justify-content:center; "
        "margin-right:{gap}px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); "
        "font-size: 10px; color: #111827; user-select:none;"
    ).format(gap=gap_px)

    # 小标签样式（索引）
    idx_css = "position:absolute; top:-14px; font-size: 10px; color:#6b7280;"

    # 为了支持 tooltip，把数值放在 title 属性里；可选显示索引数字（小）
    dots_html = []
    for i, val in enumerate(vec):
        color = _to_hex_color(float(val), vmin, vmax, cmap)
        title = f"i={i}, value={float(val):.4f}"
        idx_html = f'<div style="{idx_css}">{i}</div>' if show_index else ""
        dot_html = (
            f'<div title="{escape(title)}" '
            f'style="position:relative; {dot_base_css} background:{color};">'
            f'{idx_html}'
            "</div>"
        )
        dots_html.append(dot_html)

    # 颜色条图例（min→max）
    grad_css = (
        "height:10px; border-radius:6px; margin-top:6px; "
        f"background: linear-gradient(90deg, {_to_hex_color(vmin,vmin,vmax,cmap)}, {_to_hex_color(vmax,vmin,vmax,cmap)});"
    )
    legend_html = (
        '<div style="display:flex; align-items:center; gap:8px; margin-top:4px; font-size:12px; color:#374151;">'
        f'<div style="min-width:24px; text-align:right;">{vmin:.2f}</div>'
        f'<div style="flex:1; {grad_css}"></div>'
        f'<div style="min-width:24px;">{vmax:.2f}</div>'
        '</div>'
    )

    return f'<div style="{container_css}">' + "".join(dots_html) + "</div>" + legend_html