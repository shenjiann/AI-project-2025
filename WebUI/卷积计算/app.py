from shiny import App, render, ui, reactive, req
import numpy as np
from pathlib import Path
from utility import *
from PIL import Image

app_ui = ui.page_fluid(
    # 对ID为 threedep 的图片输出容器调整CSS，缩小下方外边距
    ui.tags.style("""
        #threedep {
            margin-bottom: -320px !important;
        }
    """),

    # 加载 MathJax
    ui.HTML("""
    <script type="text/javascript"
    id="MathJax-script"
    async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
    Shiny.addCustomMessageHandler('refresh-mathjax', function(_) {
        if (window.MathJax) {
        MathJax.typesetPromise();
        }
    });
    </script>
    """
    ),

    # 图片和标题
    ui.output_image("threedep"),
    ui.panel_title("二维卷积计算"),
    ui.layout_columns(
        ui.div(
            ui.navset_card_tab(
                ui.nav_panel(
                    "输入矩阵", 
                    ui.input_slider('height', r'\(d_H\)', 1, 10, 5),
                    ui.input_slider('width', r'\(d_W\)', 1, 10, 5),
                    value='array'
                ),
                ui.nav_panel(
                    "输入自选图片", 
                    ui.input_file(
                        'image', 
                        None, 
                        button_label='浏览文件',
                        placeholder='当前默认: Jerry.png',
                        accept='image/*'),
                    value='image',
                ),
                id='input_type'
            ),
            ui.card(
                ui.h5('卷积核设定'),
                ui.input_slider('size', r'\(f\)', 1, 7, 3, step=2),
                ui.input_slider("stride", r"\(s\)", 1, 5, 1),
                ui.input_slider('padding', r'\(p\)', 0, 5, 0),
                ui.input_select('kernel','选择卷积核类型', conv_modes),
                ui.input_numeric("seed", "随机种子", 42),
            ),
        ),
        ui.card(
            ui.h5('输出'),
            ui.output_ui('get_html_output')
        ),
        col_widths=[4, 8]
    ),
)

def server(input, output, session):
    @render.image  
    def threedep():
        here = Path(__file__).parent
        return {
            "src": here/"figs/threedep.png",
            "style": "width: 100%; max-height: 60px; display: block; margin: 0; padding: 0;"
        }

    @reactive.calc
    def get_input_array():
        if input.input_type() == 'array':
            return generate_int_array(
                height=input.height(), 
                width=input.width(), 
                seed=input.seed())
        else:
            pass
    
    @reactive.calc
    def get_input_image():
        if input.input_type() == 'image':
            if input.image() is None:
                file_path = Path(__file__).parent/'figs/Jerry.png'
            elif input.image() is not None:
                file_path = req(input.image())[0]['datapath']
            return Image.open(file_path).convert('L')
        else:
            pass

    @reactive.calc
    def get_kernel_array():
        return generate_kernel(
            kernel=input.kernel(), 
            size=input.size(), 
            seed=input.seed())

    @reactive.calc
    def get_output_array():
        if input.input_type() == 'array':
            return convolve(
                input=get_input_array(), 
                kernel=get_kernel_array(), 
                stride=input.stride(), 
                padding=input.padding()
            )
        else:
            pass
    
    @reactive.calc
    def get_output_image():
        if input.input_type() == 'image':
            output_array = convolve(
                input=np.array(get_input_image()),
                kernel=get_kernel_array(),
                stride=input.stride(),
                padding=input.padding()
            )
            output_array = (output_array - output_array.min()) * (255.0 / (output_array.max() - output_array.min()))
            return Image.fromarray(np.uint8(output_array))

    @render.ui
    def get_html_output():
        if input.input_type() == 'array':
            input_tex = mat2latex(get_input_array())
            kernel_tex = mat2latex(get_kernel_array())
            output_tex = mat2latex(get_output_array())

            # 上下标说明
            input_with_note = rf"\underset{{\text{{padding}} = {input.padding()}}}{{{input_tex}}}"
            output_with_note = rf"\underset{{\text{{stride}} = {input.stride()}}}{{{output_tex}}}"

            # 完整 LaTeX 表达式
            full_expr = r"\[" + input_with_note + r"\xrightarrow{" + r'\ast' + kernel_tex + r"}" + output_with_note + r"\]"

            return ui.HTML(f"""
            <div id="mathjax-container" style="text-align: left;">{full_expr}</div>
            <script>MathJax.typesetPromise();</script>
            """)
        elif input.input_type() == 'image':
            input_b64 = pil_to_base64(get_input_image())
            output_b64 = pil_to_base64(get_output_image())
            kernel_tex = mat2latex(get_kernel_array())
            return ui.HTML(
                get_html_img2img(
                    input_b64=input_b64, 
                    padding=input.padding(), kernel_tex=kernel_tex,
                    output_b64=output_b64,
                    stride=input.stride()
                    )
                )
        
app = App(app_ui, server)


