from shiny import App, ui, render, session
import asyncio

# UI 部分
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.script({
            "id": "MathJax-script",
            "async": "async",
            "src": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        }),
        ui.tags.script("""
            Shiny.addCustomMessageHandler("refresh-mathjax", function(message) {
                if (window.MathJax) {
                    MathJax.typesetPromise();
                }
            });
        """)
    ),
    ui.h3("测试 MathJax 渲染"),
    ui.output_ui("math_output"),
    ui.output_ui("math_output2"),

)

# server 函数
def server(input, output, session):
    pass

# 输出 HTML 含公式
@render.ui
def math_output():
    return ui.HTML(r"<div style='font-size: 20px;'>这是一个公式：\( a^2 + b^2 = c^2 \)</div>")

@render.ui
async def math_output2():
    await asyncio.sleep(0.1)
    await session().send_custom_message("refresh-mathjax", {})
    pass

# 启动应用
app = App(app_ui, server)