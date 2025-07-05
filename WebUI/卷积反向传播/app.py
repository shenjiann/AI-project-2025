import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import matrix_to_html, pad_matrix, Z, W, dZ, dZ0, dW

app_ui = ui.page_fluid(
    # 加载 CSS 和 MathJax 配置
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    
    # 顶部图片和标题
    # ui.output_image("threedep"),
    # ui.panel_title("卷积层的反向传播"),

    ui.input_slider('height', 'height', min=3, max=5, value=3, step=1),
    ui.input_slider('width', 'width', min=3, max=5, value=3, step=1),
    ui.input_slider('channel', 'channel', min=1, max=2, value=1, step=1),

    ui.h4(' '),
    ui.output_ui("Z_html"),
    ui.output_ui("W_html"),
    ui.output_ui("dZ0_html"),
    ui.card(
        ui.input_slider("row_highlight", "高亮行 (row)", min=1, max=3, value=1, step=1),
        ui.input_slider("col_highlight", "高亮列 (column)", min=1, max=3, value=1, step=1),
        ui.output_ui("dZ_html"),
    ),
    ui.h4(' '),
    ui.card(
        ui.output_ui("dW_html"),
    )
)

def server(input, output, session):

    @reactive.effect
    def _():
        # 基于input.height, input.width更新input.row_highlight, input.col_highlight的最大值
        ui.update_slider(id='row_highlight', label='高亮行 (row)', min=1, max=input.height(), value=1, step=1)
        ui.update_slider(id='col_highlight', label='高亮列 (column)', min=1, max=input.width(), value=1, step=1)

    @render.ui
    def Z_html():
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( Z^{[l-1]} = \\)</span>',
            matrix_to_html(Z),
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def W_html():
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( W^{[l]} = \\)</span>',
            matrix_to_html(W),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def dZ0_html():
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( dZ^{[l]}_0 = \\)</span>',
            matrix_to_html(dZ0),
        ]
        return ui.HTML("".join(parts))

    @render.ui
    def dZ_html():
        row = input.row_highlight() - 1
        col = input.col_highlight() - 1
        highlight_coords = [(row, col)]

        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( dZ^{[l-1]} = \\)</span>',
            matrix_to_html(dZ, highlight=highlight_coords),
            f'<span class="equation-symbol"> \\( = ({W[0,0,0,0]}) \\times \\)',
            matrix_to_html(pad_matrix(dZ0, (0,1,0,1))),
            f'<span class="equation-symbol"> \\( + ({W[0,0,0,1]}) \\times \\)',
            matrix_to_html(pad_matrix(dZ0, (1,0,0,1))),
            f'<span class="equation-symbol"> \\( + ({W[0,0,1,0]}) \\times \\)',
            matrix_to_html(pad_matrix(dZ0, (0,1,1,0))),
            f'<span class="equation-symbol"> \\( + ({W[0,0,1,1]}) \\times \\)',
            matrix_to_html(pad_matrix(dZ0, (1,0,1,0))),
            '</div>'
        ]
        return ui.HTML("".join(parts))
        
    @render.ui
    def dW_html():
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( d W^{[l]} = \\)</span>',
            matrix_to_html(dW),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    
    # --- MathJax 渲染逻辑 ---
    # 发送消息到客户端，触发 MathJax 渲染
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

    # --- 顶部图片 ---
    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0;"
        }
app = App(app_ui, server)