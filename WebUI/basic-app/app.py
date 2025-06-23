from shiny import App, render, ui
import numpy as np

app_ui = ui.page_fluid(
    ui.panel_title("Hello Shiny!"),
    ui.input_slider("n", "N", 0, 100, 20),
    ui.output_text_verbatim("txt"),
    ui.output_ui("matrix_latex"),
    
    ui.HTML("""
    <script type="text/javascript"
        id="MathJax-script"
        async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    """)
)

def server(input, output, session):
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"

    @render.ui
    def matrix_latex():
        A = np.array([[1, 2], [3, 4]])
        latex_matrix = r"\begin{bmatrix}" + \
                    r" \\\ ".join([" & ".join(map(str, row)) for row in A]) + \
                    r"\end{bmatrix}"
        latex_full = f"$$ {latex_matrix} $$"

        # 包装 MathJax + typeset 触发脚本
        return ui.HTML(f"""
        <div id="mathjax-container">{latex_full}</div>
        <script>
            MathJax.typesetPromise();
        </script>
        """)

app = App(app_ui, server)