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
    *,                 # 仅关键字参数，避免顺序歧义
    label: str,
    tensor,            # callable 或直接 tensor
    highlight=None     # callable / list / None
):
    def _get_tensor():
        return tensor() if callable(tensor) else tensor
    def _get_highlight():
        if callable(highlight):
            return highlight()
        return highlight or []

    @output
    @render.ui
    def tensor_display():
        t = _get_tensor()
        hl = _get_highlight()

        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {label} = \\)</span>',
            "\\( , \\)".join(tensor2html(t, hl)),
            '</div>'
        ]
        return ui.HTML("".join(parts))