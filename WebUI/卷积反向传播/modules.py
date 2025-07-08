from shiny import module, ui, render
from utility import overlap_tensor2html
import torch

# 自定义模块：张量展示
@module.ui
def display_tensor_ui():
    return ui.output_ui("tensor_display")

@module.server
def display_tensor_server(
    input, output, session, *,
    label,
    tensor, # callable 或直接 tensor
    overlay=True,
    highlight=None # callable / list / None
):
    @output
    @render.ui
    def tensor_display():
        lbl = label() if callable(label) else label
        t   = tensor() if callable(tensor) else tensor
        overlay_val = overlay() if callable(overlay) else overlay
        if t is None:
            return ui.HTML('')
        hl  = highlight() if callable(highlight) else (highlight or [])

        parts = [
            '<div class="equation">',
            f'<span class="equation-symbol">\\( {lbl} \\)</span>',
            overlap_tensor2html(
                tensor=t,
                overlay=overlay_val,
                highlight=hl),
            '</div>'
        ]

        return ui.HTML("".join(parts))