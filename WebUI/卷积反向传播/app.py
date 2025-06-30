from shiny import App, ui, render, reactive, Session
import numpy as np
from pathlib import Path
from utility import *
import torch
import torch.nn.functional as F

Z = torch.randint(-5, 6, (1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
W = torch.randint(-3, 4, (1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
Z0 = F.conv2d(Z, W)
dZ0 = torch.randint(-3, 3, Z0.shape, dtype=torch.float32)
Z0.backward(dZ0)
dZ = Z.grad
dW = W.grad

app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    # ui.output_image("threedep"),
    ui.panel_title("卷积反向传播"),
    ui.h4('输入矩阵'),
    ui.output_ui("Z_html"),
    ui.h4('卷积核'),
    ui.output_ui("W_html"),
    ui.h4('dZ0'),
    ui.output_ui("dZ0_html"),
    ui.h4('dZ'),
    ui.output_ui("dZ_html"),
    ui.h4('dW'),
    ui.output_ui("dW_html"),
)

def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.trigger_mathjax)
    def _():
        session.send_custom_message('refresh-mathjax', {})

    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "width: 100%; max-height: 60px; display: block; margin: 0; padding: 0;"
        }

    @output
    @render.ui
    def Z_html():
        matrix_html = matrix_to_html(Z, highlight=[(0, 0), (1, 1), (2, 2)])
        return ui.HTML(matrix_to_html(Z))
    
    @output
    @render.ui
    def W_html():
        return ui.HTML(matrix_to_html(W))
    
    @output
    @render.ui
    def dZ0_html():
        return ui.HTML(matrix_to_html(dZ0))

    @output
    @render.ui
    def dZ_html():
        return ui.HTML(matrix_to_html(dZ))

    @output
    @render.ui
    def dW_html():
        return ui.HTML(matrix_to_html(dW))

app = App(app_ui, server)