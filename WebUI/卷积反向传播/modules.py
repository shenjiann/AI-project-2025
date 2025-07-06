from shiny import module, ui, render
from utility import tensor2html
import torch

# 自定义模块：张量展示
@module.ui
def display_tensor_ui():
    return ui.output_ui("tensor_display")

@module.server
def display_tensor_server(
    input, output, session, 
    label: str, 
    data, 
    highlight = None):
    @render.ui
    def tensor_display():
        if callable(data): # 对于reactive
            tensor = data()[label]
        elif isinstance(data, torch.Tensor): # 对于tensor
            data_calc = data

        if callable(highlight): # 对于reactive
            hl = highlight()
        elif highlight is None: # 对于None
            hl = []
        else: # 对于list
            hl = highlight

        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {label} = \\)</span>',
            "\\( , \\)".join(tensor2html(tensor, hl)),
            '</div>'
        ]
        return ui.HTML("".join(parts))