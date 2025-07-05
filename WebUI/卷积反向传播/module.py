from shiny import module, ui, render
from utility import tensor2html

@module.ui
def display_tensor_ui():
    # 返回固定ID的输出容器
    return ui.output_ui("tensor_display")

@module.server
def display_tensor_server(input, output, session, label: str, data):
    @output
    @render.ui
    def tensor_display():  # 使用有效的函数名
        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {label} = \\)</span>',
            "\( , \)".join(tensor2html(data[label])),
            '</div>'
        ]
        return ui.HTML("".join(parts))