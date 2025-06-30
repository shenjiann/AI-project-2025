from shiny import App, ui, render, reactive, Session
import numpy as np
from pathlib import Path
from utility import *


app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),
    ui.HTML((Path(__file__).parent / "www/mathjax_config.html").read_text(encoding="utf-8")),
    # ui.output_image("threedep"),
    ui.panel_title("卷积反向传播"),
    ui.h4('Z'),
    ui.output_ui("Z_html"),
    ui.h4('W'),
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
        return ui.HTML(matrix_to_html(Z, highlight=[(0,0), (2,2), (0, 2)]))
    
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