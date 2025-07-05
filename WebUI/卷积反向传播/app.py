import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import matrix_to_html, pad_matrix, generate_data

app_ui = ui.page_fluid(
    # 加载 CSS 和 MathJax 配置
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    
    # 顶部图片和标题
    # ui.output_image("threedep"),
    ui.panel_title("卷积层的反向传播"),

    # 输入和卷积核设定
    ui.card(
        ui.card_header(r"输入 \( Z^{[l-1]} \) 和卷积核 \(W\)设定"),
        ui.layout_sidebar(
            # 侧边栏
            ui.sidebar(
                ui.input_slider('height', r'\( d_H^{[l-1]} \)', min=3, max=5, value=3, step=1),
                ui.input_slider('width', r'\( d_W^{[l-1]} \)', min=3, max=5, value=3, step=1),
                ui.input_slider('channel', r'\( d_C^{[l-1]} \)', min=1, max=2, value=1, step=1),
                ui.input_slider('size', r'\( f^{[l]} \)', min=2, max=3, value=2, step=1),
                r'\( d_C^{[l]} = 1\)',
            ),
        ui.output_ui('Z_display'),
        ui.output_ui('W_display'),
        ui.output_ui('Z0_display'),
        ui.output_ui('dZ0_display'),
        ),
    ),  

    # 输入梯度展示
    ui.card(
        ui.card_header(r"输入梯度 \( dZ^{[l-1]} \)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button('dZnext', '下一步'),
                ui.input_action_button('dZreset', '重置')
            ),
            ui.output_ui("dZ_display")
        ),
    ),

    # 卷积核梯度展示
    ui.card(
        ui.card_header(r"卷积核梯度 \( d W^{[l]} \)"),
        ui.output_ui("dW_html"),
    )
)

def server(input, output, session):

    @reactive.calc
    def data():
        return generate_data(
            d_H=input.height(),
            d_W=input.width(),
            d_C=input.channel(),
            f=input.size(),
            seed=42
        )
    

    @reactive.effect
    def _():
        # 基于input.height, input.width更新input.row_highlight, input.col_highlight的最大值
        ui.update_slider(id='row_highlight', label='高亮行 (row)', min=1, max=input.height(), value=1, step=1)
        ui.update_slider(id='col_highlight', label='高亮列 (column)', min=1, max=input.width(), value=1, step=1)

    @render.ui
    def Z_display():
        Z, _, _, _, _, _ = data()
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( Z^{[l-1]} = \\)</span>',
            matrix_to_html(Z),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def W_display():
        _, W, _, _, _, _ = data()
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( W^{[l]} = \\)</span>',
            matrix_to_html(W),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def Z0_display():
        _, _, Z0, _, _, _ = data()
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( Z_0^{[l]} = \\)</span>',
            matrix_to_html(Z0),
            '</div>'
        ]
        return ui.HTML("".join(parts))

    @render.ui
    def dZ0_display():
        _, _, _, dZ0, _, _ = data()
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">\\( dZ^{[l]}_0 = \\)</span>',
            matrix_to_html(dZ0),
            '</div>'
        ]
        return ui.HTML("".join(parts))

    @render.ui
    def dZ_display():
        _, W, _, dZ0, dZ, _ = data()
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
    def dW_display():
        _, _, _, _, _, dW = data()
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
    @reactive.event(input.height, input.width, input.channel, input.size)
    async def _slider_mathjax_render():
        await trigger_mathjax_render_on_client()

    # --- 顶部图片 ---
    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0;"
        }
    
    @render.image
    def conv():
        return {
            'src': Path(__file__).parent/"www/conv.png",
            'styles': "width: 100%; height: auto; max-width: 500px; margin: auto; display: block;"
        }
app = App(app_ui, server)