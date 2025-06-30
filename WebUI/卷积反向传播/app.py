from shiny import App, ui, render, reactive
import numpy as np
from pathlib import Path
from utility import *


# 生成一个 5×5 的整数矩阵
input_mat = np.random.randint(0, 10, size=(5, 5))
kernel_mat = np.random.randint(-5, 5, size=(3, 3))


app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    # ui.output_image("threedep"),
    ui.panel_title("卷积反向传播"),
    ui.h4('输入矩阵'),
    ui.output_ui("input_matrix"),
    ui.h4('卷积核'),
    ui.output_ui("kernel_matrix"),

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
    )
)

def server(input, output, session):
    @render.image  
    def threedep():
        return {
            "src": Path(__file__).parent/"figs/threedep.png",
            "style": "width: 100%; max-height: 60px; display: block; margin: 0; padding: 0;"
        }

    @output
    @render.ui
    def input_matrix():
        matrix_html = matrix_to_html(input_mat, highlight=[(0, 0), (1, 1), (2, 2)])
        return ui.HTML(rf"""
            <div class="matrix-row">
                <div class="matrix-label">\\[ Z_0 = \\]</div>
                {matrix_html}
            </div>
            <script>Shiny.setInputValue('trigger-mathjax', Math.random());</script>
        """)
    
    @output
    @render.ui
    def kernel_matrix():
        return ui.HTML(matrix_to_html(kernel_mat))
    
    @reactive.Effect
    @reactive.event(input.trigger_mathjax)
    def _():
        session.send_custom_message('refresh-mathjax', {})

app = App(app_ui, server)