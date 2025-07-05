import asyncio
from shiny import App, ui, render, reactive
from shiny.session import get_current_session
from pathlib import Path

app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),

    ui.h2("HTML 和 MathJax 渲染示例"),
    ui.hr(),

    ui.h4("示例 1: 纯文本和行内公式"),
    ui.output_ui("example1"),
    ui.h4("示例 2: 块级公式"),
    ui.output_ui("example2"),
    ui.h4("示例 3: HTML 块级元素与公式"),
    ui.output_ui("example3"),
    ui.h4("示例 4: HTML 行内元素与公式"),
    ui.output_ui("example4"),
    ui.h4("示例 5: HTML 行内块级元素与公式"),
    ui.output_ui("example5"),
)

def server(input, output, session):

    # MathJax 渲染逻辑
    async def trigger_mathjax_render_on_client():
        session = get_current_session()
        await asyncio.sleep(0.001) # Give browser a moment to update DOM
        await session.send_custom_message("render-mathjax", {})

    @reactive.effect
    @reactive.event(input.session_initialized_client)
    async def _initial_mathjax_render():
        await trigger_mathjax_render_on_client()

    @render.ui
    def example1():
        # 纯文本和行内公式
        return ui.HTML(r"""
            <p>这是一段包含行内公式的文本：当 \(a \ne 0\), 则 \(ax^2 + bx + c = 0\) 的解为
            \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
            这是公式后的文本。</p>
        """)

    @render.ui
    def example2():
        # 块级公式
        return ui.HTML(r"""
            <p>下面的公式将单独占一行：</p>
            \[ E=mc^2 \]
            <p>这是公式后的文本。</p>
        """)

    @render.ui
    def example3():
        # HTML 块级元素与公式
        return ui.HTML(r"""
            <div class="my-block-div">
                这是一个 div 块。
                其中包含一个行内公式：\(y = x^2\).
            </div>
            <p>另一个段落。</p>
            <div class="my-block-div">
                <p>内部的段落。</p>
                <p>块级公式：</p>
                \[ \sum_{i=1}^n i = \frac{n(n+1)}{2} \]
            </div>
        """)

    @render.ui
    def example4():
        # HTML 行内元素与公式
        return ui.HTML(r"""
            <span>这是一个 span 行内元素。</span>
            <span>行内公式：\( \int_a^b f(x) dx \).</span>
            <span>另一个 span 元素。</span>
            <p>这个段落后面的行内公式：\( \alpha + \beta \).</p>
        """)

    @render.ui
    def example5():
        # HTML 行内块级元素与公式
        return ui.HTML(r"""
            <div class="my-inline-block-div">
                行内块级 div 1.
                \( A \times B \)
            </div>
            <div class="my-inline-block-div">
                行内块级 div 2.
                \[ \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} \]
            </div>
            <div class="my-inline-block-div">
                行内块级 div 3.
            </div>
            <p>这是一个正常的段落，来观察上面 div 的布局。</p>
        """)

app = App(app_ui, server)