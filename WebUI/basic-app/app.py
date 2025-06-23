from shiny import App, render, ui, reactive
import numpy as np
from pathlib import Path
from utility import kernel_choices, generate_kernel, mat_to_latex, generate_input, convolve

app_ui = ui.page_fluid(
    # 对ID为 threedep 的图片输出容器调整CSS，缩小下方外边距
    ui.tags.style("""
        #threedep {
            margin-bottom: -320px !important;
        }
    """),

    # 图片和标题
    ui.output_image("threedep"),
    ui.panel_title("二维卷积计算"),
    ui.layout_columns(
        ui.card(
            ui.h5('参数设定'),
            ui.input_slider('height', r'\( d_H \)', 1, 10, 5),
            ui.input_slider('width', r'\(d_W\)', 1, 10, 5),
            ui.input_slider('size', r'\(f\)', 1, 7, 3, step=2),
            ui.input_slider("stride", r"\(s\)", 1, 5, 1),
            ui.input_slider('padding', r'\(p\)', 0, 5, 0),
            ui.input_select('kernel','选择卷积核类型', kernel_choices),
            ui.input_numeric("seed", "随机种子", 42),
            width=1
        ),
        ui.card(
            ui.output_ui('input2output_latex'),
            width=4
        )
    ),
    
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
    """)
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
    def input_array():
        return generate_input(height=input.height(), width=input.width(), seed=input.seed())

    @reactive.calc
    def kernel_array():
        return generate_kernel(input.kernel(), input.size(), seed=input.seed())

    @reactive.calc
    def output_array():
        return convolve(input=input_array(), kernel=kernel_array(), stride=input.stride(), padding=input.padding())
    
    @render.ui
    def input2output_latex():
        input_tex = mat_to_latex(input_array(), wrap=False)
        kernel_tex = mat_to_latex(kernel_array(), wrap=False)
        output_tex = mat_to_latex(output_array(), wrap=False)
        padding_val = input.padding()
        stride_val = input.stride()

        # 上下标说明
        input_with_note = rf"\underset{{\text{{padding}} = {padding_val}}}{{{input_tex}}}"
        output_with_note = rf"\underset{{\text{{stride}} = {stride_val}}}{{{output_tex}}}"

        # 完整 LaTeX 表达式
        full_expr = r"\[" + input_with_note + r"\xrightarrow{" + kernel_tex + r"}" + output_with_note + r"\]"

        return ui.HTML(f"""
        <div id="mathjax-container" style="text-align: left;">{full_expr}</div>
        <script>MathJax.typesetPromise();</script>
        """)


    

    
app = App(app_ui, server)


