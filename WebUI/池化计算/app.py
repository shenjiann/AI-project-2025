from shiny import App, render, ui, reactive
import numpy as np
from pathlib import Path
from utility import pool_modes, generate_input, pool, mat2latex

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

    ui.output_image("threedep"),
    ui.panel_title("二维池化计算"),

    ui.layout_columns(
        ui.card(
            ui.h5('参数设置'),
            ui.input_slider('height', r'\(d_H\)', 1, 10, 5),
            ui.input_slider('width', r'\(d_W\)', 1, 10, 5),
            ui.input_slider("size", r"\(f\)", 1, 7, 2),
            ui.input_slider("stride", r"\(s\)", 1, 5, 2),
            ui.input_slider("padding", r"\(p\)", 0, 5, 0),
            ui.input_select("mode", "池化方式", pool_modes),
            ui.input_numeric("seed", "随机种子", 42),
        ),

        ui.card(
            ui.h5('输出'),
            ui.output_ui('input2output_latex'),
        )
    )
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
        return generate_input(height=input.height(),
                              width=input.width(),
                              seed=input.seed())

    @reactive.calc
    def output_array():
        return pool(input_array=input_array(),
                    kernel_size=input.size(),
                    stride=input.stride(),
                    padding=input.padding(),
                    mode=input.mode())
    
    @render.ui
    def input2output_latex():
        input_tex = mat2latex(input_array(), wrap=False)
        output_tex = mat2latex(output_array(), wrap=False)

        arrow_tex = rf"{input.mode()}\;{input.size()}\times{input.size()}"
        input_note = rf"\underset{{\text{{padding}} = {input.padding()}}}{{{input_tex}}}"
        output_note = rf"\underset{{\text{{stride}} = {input.stride()}}}{{{output_tex}}}"
        full_expr = r"\[" + input_note + r"\xrightarrow{" + arrow_tex + r"}" + output_note + r"\]"
        return ui.HTML(
            f"""
            <div style=\"text-align: left;\">{full_expr}</div>
            <script>MathJax.typesetPromise();</script>
            """
        )
app = App(app_ui, server)