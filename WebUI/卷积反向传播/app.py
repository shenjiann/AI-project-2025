import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import tensor2html
from modules import display_tensor_ui, display_tensor_server
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
            ),
            # 主面板
            display_tensor_ui('Z_block'),
            display_tensor_ui('W_block'),
            display_tensor_ui('Z0_block'),
            display_tensor_ui('dZ0_block'),
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
                # 主面板
                display_tensor_ui('dZ_block'),
                ui.h6('计算过程：'),
                ui.output_ui('steps'),
                ui.output_ui('hl'),
                display_tensor_ui('Zslice_block'),
                ui.output_ui('Z0ij_block'),
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
                display_tensor_ui('dW_block'),
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


    # --- 输入梯度计算逻辑 ---
    current_step_dZ = reactive.Value(0)

        # 监测下一步
    @reactive.effect
    @reactive.event(input.dZnext)
    def _next_step_dZ():
        elems_per_channel  = data().d_H_l * data().d_W_l
        if current_step_dZ() < elems_per_channel:
            current_step_dZ.set(current_step_dZ() + 1)

    @render.ui # 监控steps
    def steps():
        return current_step_dZ()
    
    @render.ui # 监控高亮坐标
    def hl():
        return data().get_focus_coords(current_step_dZ())

    # --- tensor展示 ---
    @render.ui
    def dims_l():
        return ui.HTML(
            fr"\(d_H^{{[l]}} = {data().d_H_l},\ d_W^{{[l]}} = {data().d_W_l},\ d_C^{{[l]}} = {data().d_C_l}\)"
        )
    
    display_tensor_server(id='Z_block', label='Z^{[l-1]}', tensor=lambda: data().Z)
    display_tensor_server(id='W_block', label='W^{[l]}', tensor=lambda: data().W)
    display_tensor_server(id='Z0_block', label='Z_0^{[l]}', tensor=lambda: data().Z0)
    display_tensor_server(id='dZ0_block', label='dZ_0^{[l]}', tensor=lambda: data().dZ0, highlight=lambda:data().get_focus_coords(current_step_dZ()))
    display_tensor_server(id='dZ_block', label='dZ^{[l-1]}', tensor=lambda: data().dZ)
    display_tensor_server(id='dW_block', label='dW^{[l]}', tensor=lambda: data().dW)
    
    display_tensor_server(id='Zslice_block', label='Z_{slice,i,j}^{[l]}', tensor=lambda: data().get_Z_slice_ij(current_step_dZ()))
    display_tensor_server(id='Z0ij_block', label='Z_{0,i,j}^{[l]}', tensor=lambda: data().get_Z0_ij(current_step_dZ()))
    


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