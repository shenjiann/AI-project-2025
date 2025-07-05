from shiny import module, ui, render
from typing import Callable
from utility import tensor2html

@module.ui
def display_tensor_ui():
    # 返回固定ID的输出容器
    return ui.output_ui("tensor_display")

@module.server
def display_tensor_server(input, output, session, label, data_calc):
    @render.ui
    def tensor_display():  # 使用有效的函数名
        tensor = data_calc()[label]
        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {label} = \\)</span>',
            "\( , \)".join(tensor2html(tensor)),
            '</div>'
        ]
        return ui.HTML("".join(parts))