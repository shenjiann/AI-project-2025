import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import matrix_to_html, Z, W, dZ, dZ0, dW # Assuming these are correctly defined

app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),

    ui.input_slider('height', 'height', min=3, max=5, value=3, step=1),
    ui.input_slider('width', 'width', min=3, max=5, value=3, step=1),
    ui.input_slider('channel', 'channel', min=1, max=2, value=1, step=1),

    ui.h4(' '),
    ui.output_ui("Z_html"),
    ui.h4(' '),
    ui.output_ui("W_html"),
    ui.h4(' '),
    ui.output_ui("dZ0_html"),
    ui.h4(' '),
    ui.input_slider("row_highlight", "高亮行 (row)", min=1, max=3, value=1, step=1),
    ui.input_slider("col_highlight", "高亮列 (column)", min=1, max=3, value=1, step=1),
    ui.output_ui("dZ_html"),
    ui.h4(' '),
    ui.output_ui("dW_html"),
)

def server(input, output, session):

    @reactive.effect
    def _():
        # Update slider max values based on input
        ui.update_slider(id='row_highlight', label='高亮行 (row)', min=1, max=input.height(), value=1, step=1)
        ui.update_slider(id='col_highlight', label='高亮列 (column)', min=1, max=input.width(), value=1, step=1) # Also update col_highlight

    @render.ui
    def Z_html():
        return ui.HTML(matrix_to_html(Z, highlight=[(0, 0), (2, 2), (0, 2)], prefix='\\( Z = \\)'))
    
    @render.ui
    def W_html():
        out = ui.HTML(
            matrix_to_html(W, prefix="\\( W = \\) ") +
            matrix_to_html(W, prefix=' + ', highlight=[(0,0)])
            )
        return out
    
    @render.ui
    def dZ0_html():
        return ui.HTML(matrix_to_html(dZ0, prefix='\\( dZ_0 = \\) '))

    @render.ui
    def dZ_html():
        row = input.row_highlight() - 1
        col = input.col_highlight() - 1
        highlight_coords = [(row, col)]
        html = ui.HTML(r'\( dZ \) = ' + matrix_to_html(dZ, highlight=highlight_coords) + r' + ' + matrix_to_html(W))
        return html # Just return HTML, MathJax trigger is separate
        
    @render.ui
    def dW_html():
        return ui.HTML(r'\( d W^{l-1} \) = ' + matrix_to_html(dW))
    
    # --- MathJax 渲染逻辑 ---
    # 发送自定义消息到客户端，触发 MathJax 渲染
    async def trigger_mathjax_render_on_client():
        session = get_current_session()
        await asyncio.sleep(0.001)
        await session.send_custom_message("render-mathjax", {})

    # 初次加载
    @reactive.effect
    @reactive.event(input.session_initialized_client)
    async def _initial_mathjax_render():
        await trigger_mathjax_render_on_client()

    # 监听滑块变化
    @reactive.effect
    @reactive.event(input.row_highlight, input.col_highlight)
    async def _slider_mathjax_render():
        await trigger_mathjax_render_on_client()

app = App(app_ui, server)