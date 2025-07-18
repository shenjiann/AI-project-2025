import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path
from utility import overlap_tensor2html
from Data import Data
from DZCalculator import DZCalculator
from DWCalculator import DWCalculator
from modules import display_tensor_ui, display_tensor_server

app_ui = ui.page_fluid(
    # 加载 CSS 和 MathJax 配置
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    
    # 顶部图片和标题
    ui.output_image("threedep"),
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
                class_="custom-sidebar",
            ),
            # 主面板
            display_tensor_ui('Z_display'),
            display_tensor_ui('W_display'),
            display_tensor_ui('Z0_display'),
            display_tensor_ui('dZ0_display'),
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
                    class_="custom-sidebar",
                ),
                # 主面板，条件渲染
                ui.output_ui('dZ_calc_steps')
            ),
            value = 'dZ',
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
                    class_="custom-sidebar",
                ),
                # 主面板
                ui.output_ui('dW_calc_steps'),
            ),
            value = 'dW'
        ),
        id = 'nav_tab',
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

    @reactive.calc
    def dZcalc():
        return DZCalculator(data())
    
    @reactive.calc
    def dWcalc():
        return DWCalculator(data())
    
    # --- dZ梯度计算逻辑 ---
    # 监测下一步
    @reactive.effect
    @reactive.event(input.dZnext)
    def _():
        # 尝试增加 current_step_dZ，setter 会控制其最大值
        dZcalc().steps = dZcalc().steps + 1
        # 只有当最后一个切片尚未累加时，才执行累加逻辑
        if dZcalc().steps >= 1 and not dZcalc().last_slice_accumulated:
            dZ_slice_val = dZcalc().get_dZ_slice_ij()
            if dZ_slice_val is not None:
                dZcalc().add_to_dZ_cache(dZ_slice_val)

    @reactive.effect
    @reactive.event(input.dWnext)
    def _():
        # 尝试增加 current_step_dZ，setter 会控制其最大值
        dWcalc().steps = dWcalc().steps + 1
        # 只有当最后一个切片尚未累加时，才执行累加逻辑
        if dWcalc().steps >= 1 and not dWcalc().last_slice_accumulated:
            dW_slice_val = dWcalc().get_dW_slice_ij()
            if dW_slice_val is not None:
                dWcalc().add_to_dW_cache(dW_slice_val)
    
    @reactive.effect
    @reactive.event(input.dZreset)
    def _():
        dZcalc().steps = 0
        dZcalc().reset_dZ_cache() # 重置 dZ_cache

    @reactive.effect
    @reactive.event(input.dWreset)
    def _():
        dWcalc().steps = 0
        dWcalc().reset_dW_cache() # 重置 dZ_cache


    # --- 所有展示的公式 ---
    @render.ui
    def dims_l():
        return ui.HTML(
            fr"\(d_H^{{[l]}} = {data().d_H_l},\ d_W^{{[l]}} = {data().d_W_l},\ d_C^{{[l]}} = {data().d_C_l}\)"
        )
    
    display_tensor_server(
        id='Z_display',
        label='Z^{[l-1]} = ',
        tensor=lambda: data().Z,
        overlay=lambda: input.overlay(),
        highlight=lambda: dZcalc().get_highlight_Z() if input.nav_tab() == 'dZ' else dWcalc().get_highlight_Z(),
    )
    
    display_tensor_server(
        id='W_display',
        label=' W^{[l]} = ',
        tensor=lambda: data().W,
        overlay=lambda: input.overlay()
    )

    display_tensor_server(
        id='Z0_display',
        label=' Z_0^{[l]} = ',
        tensor=lambda: data().Z0,
        overlay=lambda: input.overlay(),
        highlight=lambda: dZcalc().get_highlight_Z0() if input.nav_tab() == 'dZ' else dWcalc().get_highlight_Z0(),
    )

    display_tensor_server(
        id='dZ0_display',
        label=' dZ_0^{[l]} = ',
        tensor=lambda: data().dZ0,
        overlay=lambda: input.overlay(),
        highlight=lambda: dZcalc().get_highlight_Z0() if input.nav_tab() == 'dZ' else dWcalc().get_highlight_Z0(),
    )

    display_tensor_server(
        id='dZ_display',
        label=' dZ^{[l-1]} = ',
        tensor=lambda: data().dZ,
        overlay=lambda: input.overlay()
    )

    display_tensor_server(
        id='dW_display',
        label=' dW^{[l]} = ',
        tensor=lambda: data().dW,
        overlay=lambda: input.overlay()
    )
    
    @render.ui
    def dZ_calc_s1():
        if input.nav_tab() != 'dZ':
            return None
        parts = [
            '<div class="equation">',
            '注意到',
            '<span class="equation-symbol">',
            rf'\( Z_{{0,{dZcalc().get_focus_i()+1},{dZcalc().get_focus_j()+1}}}^{{[l]}} = \)', 
            rf'\( Z_{{slice,{dZcalc().get_focus_i()+1},{dZcalc().get_focus_j()+1}}}^{{[l-1]}} \ast \)',
            r'\( W = \) </span>',
            overlap_tensor2html(
                tensor=dZcalc().get_Z_slice_ij(),
                overlay=input.overlay()),
            r'<span class="equation-symbol"> \( \ast \) </span>',
            overlap_tensor2html(
                tensor=data().W,
                overlay=input.overlay()
            ),
            r'<span class="equation-symbol"> \( = \) </span>',
            overlap_tensor2html(
                tensor=dZcalc().get_Z0_ij(),
                overlay=input.overlay()
            ),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def dZ_calc_s2():
        if input.nav_tab() != 'dZ':
            return None
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">',
            rf'\( dZ_{{slice,{dZcalc().get_focus_i()+1},{dZcalc().get_focus_j()+1}}}^{{[l-1]}} = \)',
            rf'\( dZ_{{0,{dZcalc().get_focus_i()+1},{dZcalc().get_focus_j()+1}}}^{{[l]}} \times \)',
            r'\( W = \) </span>',
            '</span>',
            overlap_tensor2html(
                tensor=dZcalc().get_dZ0_ij(),
                overlay=input.overlay()),
            r'<span class="equation-symbol"> \( \times \) </span>',
            overlap_tensor2html(
                tensor=data().W,
                overlay=input.overlay()
            ),
            r'<span class="equation-symbol"> \( = \) </span>',
            overlap_tensor2html(
                tensor=dZcalc().get_dZ_slice_ij(),
                overlay=input.overlay()
            ),
            '</div>'
        ]
        return ui.HTML("".join(parts))

    @render.ui
    def dZ_calc_s3():
        if input.nav_tab() != 'dZ':
            return None
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol"> \\( dZ_{cache}^{[l-1]} = \\) </span>',
            overlap_tensor2html(
                tensor=dZcalc().dZ_cache_last,
                overlay=input.overlay(),
                highlight=dZcalc().get_highlight_Z()
            ),
            '<span class="equation-symbol"> + </span>',
            rf'\( dZ_{{slice,{dZcalc().get_focus_i()+1},{dZcalc().get_focus_j()+1}}}^{{[l-1]}} = \)',
            overlap_tensor2html(
                tensor=dZcalc().dZ_cache,
                overlay=input.overlay(),
                highlight=dZcalc().get_highlight_Z()
            ),
            '</div>'
        ]
        return ui.HTML("".join(parts))


    @render.ui
    def dZ_calc_steps():
        if dZcalc().steps == 0:
            return ui.TagList(
                display_tensor_ui('dZ_display'),
                ui.HTML('<div> 点击下一步查看计算过程 </div>')
            )
        else: # 否则，展示所有计算过程相关的组件
            return ui.TagList(
                ui.output_ui('dZ_calc_s1'),
                ui.output_ui('dZ_calc_s2'),
                ui.output_ui('dZ_calc_s3'),
            )
        
    # dW计算相关

    @render.ui
    def dW_calc_s1():
        if input.nav_tab() != 'dW':
            return None
        parts = [
            '<div class="equation">',
            '注意到',
            '<span class="equation-symbol">',
            rf'\( Z_{{0,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l]}} = \)', 
            rf'\( Z_{{slice,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l-1]}} \ast \)',
            r'\( W = \) </span>',
            overlap_tensor2html(
                tensor=dWcalc().get_Z_slice_ij(),
                overlay=input.overlay()),
            r'<span class="equation-symbol"> \( \ast \) </span>',
            overlap_tensor2html(
                tensor=data().W,
                overlay=input.overlay()
            ),
            r'<span class="equation-symbol"> \( = \) </span>',
            overlap_tensor2html(
                tensor=dWcalc().get_Z0_ij(),
                overlay=input.overlay()
            ),
            '</div>'
        ]
        return ui.HTML("".join(parts))
    
    @render.ui
    def dW_calc_s2():
        if input.nav_tab() != 'dW':
            return None
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol">',
            rf'\( dW_{{slice,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l]}} = dZ_{{0,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l]}} \times Z_{{slice,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l-1]}} = \) </span>',
            overlap_tensor2html(
                tensor=dWcalc().get_dZ0_ij(),
                overlay=input.overlay()),
            r'<span class="equation-symbol"> \( \times \) </span>',
            overlap_tensor2html(
                tensor=dWcalc().get_Z_slice_ij(),
                overlay=input.overlay()),
            r'<span class="equation-symbol"> \( = \) </span>',
            overlap_tensor2html(
                tensor=dWcalc().get_dW_slice_ij(),
                overlay=input.overlay()),
            '</div>'
        ]
        return ui.HTML("".join(parts))


    @render.ui
    def dW_calc_s3():
        if input.nav_tab() != 'dW':
            return None
        parts = [
            '<div class="equation">',
            '<span class="equation-symbol"> \\( dW_{cache}^{[l]} = \\) </span>',
            overlap_tensor2html(
                tensor=dWcalc().dW_cache_last,
                overlay=input.overlay(),
                # highlight=dZcalc().get_highlight_Z()
            ),
            '<span class="equation-symbol"> + </span>',
            rf'\( dW_{{slice,{dWcalc().get_focus_i()+1},{dWcalc().get_focus_j()+1}}}^{{[l]}} = \)',
            overlap_tensor2html(
                tensor=dWcalc().dW_cache,
                overlay=input.overlay(),
                # highlight=dWcalc().get_highlight_Z()
            ),
            '</div>',
        ]
        return ui.HTML("".join(parts))

    @render.ui
    def dW_calc_steps():
        if dWcalc().steps == 0:
            return ui.TagList(
                display_tensor_ui('dW_display'),
                ui.HTML('<div> 点击下一步查看计算过程 </div>')
            )
        else: # 否则，展示所有计算过程相关的组件
            return ui.TagList(
                ui.output_ui('dW_calc_s1'),
                ui.output_ui('dW_calc_s2'),
                ui.output_ui('dW_calc_s3'),
            )
            pass



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
    @reactive.event(input.height, input.width, input.channel, input.size, input.seed, input.dZnext, input.dZreset, input.overlay, input.page_click)
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