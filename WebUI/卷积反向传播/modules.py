from shiny import module, ui, render
from utility import tensor2html

# 自定义模块：张量展示
@module.ui
def display_tensor_ui():
    return ui.output_ui("tensor_display")

@module.server
def display_tensor_server(
    input, output, session, 
    label: str, 
    data_calc, 
    highlight_calc = None):
    @render.ui
    def tensor_display():
        tensor = data_calc()[label]
        if callable(highlight_calc):
            hl = highlight_calc()
        elif highlight_calc is None:
            hl = []
        else: 
            hl = highlight_calc

        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {label} = \\)</span>',
            "\\( , \\)".join(tensor2html(tensor, hl)),
            '</div>'
        ]
        return ui.HTML("".join(parts))