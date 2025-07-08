import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import overlap_tensor2html
from Data import Data

app_ui = ui.page_fluid(
    # 加载 CSS 和 MathJax 配置
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    
    # 顶部图片和标题
    # ui.output_image("threedep"),
    ui.panel_title("卷积层的反向传播"),

    # --- 输入和卷积核设定 ---
    ui.card(
        ui.card_header(r"输入 \( Z^{[l-1]} \) 和卷积核 \(W\)设定"),
        ui.layout_sidebar(
            # 侧边栏
            ui.sidebar(
                ui.input_slider('height', r'\( d_H^{[l-1]} \)', min=3, max=5, value=3, step=1),
                ui.input_slider('width', r'\( d_W^{[l-1]} \)', min=3, max=5, value=3, step=1),
                ui.input_slider('channel', r'\( d_C^{[l-1]} \)', min=1, max=2, value=1, step=1),
                ui.input_slider('size', r'\( f^{[l]} \)', min=2, max=3, value=2, step=1),
                ui.output_ui("dims_l"),
                ui.input_numeric("seed", "随机种子", 42),
                ui.input_checkbox("overlay", "重叠显示", True),
            ),
            # 主面板
            ui.output_ui("Z_display"),
        ),
    ),  

    ui.navset_card_tab(
        # --- 输入梯度展示 ---
        ui.nav_panel(r"输入梯度 \( dZ^{[l-1]} \)",
            # 侧边栏布局
            ui.layout_sidebar(
                # 侧边栏
                ui.sidebar(
                    ui.input_action_button('dZnext', '下一步'),
                    ui.input_action_button('dZauto', '自动播放'),
                    ui.input_action_button('dZreset', '重置'),
                ),
                # 主面板，条件渲染
                ui.TagList( 
                    ui.output_ui('dynamic_calculation_details')
                ),
            ),
        ),


        # --- 卷积核梯度展示 ---
        ui.nav_panel(r"卷积核梯度 \( d W^{[l]} \)", 
            # 侧边栏布局
            ui.layout_sidebar(
                # 侧边栏
                ui.sidebar(
                    ui.input_action_button('dWnext', '下一步'),
                    ui.input_action_button('dWauto', '自动播放'),
                    ui.input_action_button('dWreset', '重置'),
                ),
                # 主面板
                ui.h6('计算过程：'),
            ),
        ),
    ),
)

def server(input, output, session):

    # --- 数据生成 ---
    @reactive.calc
    def data():
        return Data(
            d_H = input.height(),
            d_W = input.width(),
            d_C = input.channel(),
            f = input.size(),
            seed = input.seed()
        )

    # --- dZ梯度计算逻辑 ---
    # 监测下一步
    @reactive.effect
    @reactive.event(input.dZnext)
    def _():
        data().current_step_dZ = data().current_step_dZ + 1

    @reactive.effect
    @reactive.event(input.dZreset)
    def _():
        data().current_step_dZ = 0

    @render.ui # 监控steps
    def steps():
        return data().current_step_dZ
    
    @render.ui # 监控高亮坐标
    def hl():
        return data().get_focus_ij()

    # --- 所有展示的公式 ---
    @render.ui
    def dims_l():
        return ui.HTML(
            fr"\(d_H^{{[l]}} = {data().d_H_l},\ d_W^{{[l]}} = {data().d_W_l},\ d_C^{{[l]}} = {data().d_C_l}\)"
        )
    
    @render.ui
    def Z_display():
        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( Z = \\)</span>',
            overlap_tensor2html(
                data().Z,
                overlay=input.overlay()),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def dynamic_calculation_details():
        if data().current_step_dZ == 0:
            return ui.HTML('<div> 点击下一步查看计算过程 </div>')
        
        # 否则，展示所有计算过程相关的组件
        return ui.TagList(
            ui.h6('计算过程：'),
        )

    # --- MathJax 渲染逻辑 ---
    # 发送消息到客户端，触发 MathJax 渲染
    async def trigger_mathjax_render_on_client():
        session = get_current_session()
        await asyncio.sleep(0.001)
        await session.send_custom_message("render-mathjax", {})

    # 监听初次加载
    @reactive.effect
    @reactive.event(input.session_initialized_client)
    async def _initial_mathjax_render():
        await trigger_mathjax_render_on_client()

    # 监听输入变化
    @reactive.effect
    @reactive.event(input.height, input.width, input.channel, input.size, input.seed, input.dZnext)
    async def _slider_mathjax_render():
        await trigger_mathjax_render_on_client()

    # --- 图片 ---
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