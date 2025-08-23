from shiny import App, render, ui
from pathlib import Path
from torchvision import datasets, transforms


app_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent/"www/styles.css"),

    ui.output_image("threedep"),

    ui.panel_title("Hello Shiny!"),
    ui.input_slider("n", "N", 0, 100, 20),
)


def server(input, output, session):
    # --- 图片 ---
    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0;"
        }


app = App(app_ui, server)
