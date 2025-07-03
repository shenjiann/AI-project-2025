import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
# import numpy as np
from pathlib import Path
from utility import matrix_to_html, Z, W, dZ, dZ0, dW


app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),

    # ui.output_image("threedep"),
    # ui.panel_title("卷积反向传播"),
    ui.h4(' '),
    ui.output_ui("Z_html"),
    ui.h4(' '),
    ui.output_ui("W_html"),
    ui.h4(' '),
    ui.output_ui("dZ0_html"),
    ui.h4(' '),
    ui.input_slider('height', 'height', min=3, max=5, value=3, step=1),
    ui.input_slider("row_highlight", "高亮行 (row)", min=1, max=3, value=1, step=1),
    ui.input_slider("col_highlight", "高亮列 (column)", min=1, max=3, value=1, step=1),
    ui.output_ui("dZ_html"),
    ui.h4(' '),
    ui.output_ui("dW_html"),
)

def server(input, output, session):
    async def refresh_mathjax():
        """发送消息触发 MathJax 重新渲染"""
        await asyncio.sleep(0.01)  # 等待 DOM 更新
        session = get_current_session()
        await session.send_custom_message("refresh-mathjax", {})

    @reactive.effect
    def _():
        ui.update_slider(id='row_highlight', label='高亮行 (row)', min=1, max=input.height(), value=1, step=1)

    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0; max-width: none; height: auto;"
        }

    @render.ui
    def Z_html():
        # await refresh_mathjax()
        return ui.HTML(
            matrix_to_html(Z, highlight=[(0, 0), (2, 2), (0, 2)], prefix='\\( Z = \\)'))
    
    @render.ui
    def W_html():
        out = ui.HTML(
            matrix_to_html(W, prefix="\\( W = \\) ") +
            matrix_to_html(W, prefix=' + ', highlight=[(0,0)])
            )
        return out
    
    @render.ui
    def dZ0_html():
        return ui.HTML(
            matrix_to_html(dZ0, prefix='\\( dZ_0 = \\) ')
            )

    @render.ui
    async def dZ_html():
        row = input.row_highlight() - 1
        col = input.col_highlight() - 1
        highlight_coords = [(row, col)]
        html = ui.HTML(r'\( dZ \) = ' + matrix_to_html(dZ, highlight=highlight_coords) + r' + ' + matrix_to_html(W))
        return html
        
    @render.ui
    def dW_html():
        # await refresh_mathjax()
        return ui.HTML(
            r'\( d W^{l-1} \) = ' + 
            matrix_to_html(dW)
            )
    
# 使用reactive.Calc确保会话完全就绪
    @reactive.Calc
    async def init_mathjax():
        await asyncio.sleep(0.1)  # 确保会话完全初始化
        return True

    @reactive.effect
    @reactive.event(input.init_mathjax)
    async def _():
        await refresh_mathjax()
app = App(app_ui, server)