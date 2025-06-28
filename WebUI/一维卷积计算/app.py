from shiny import App, ui, render, reactive
import numpy as np

input_values = np.array([3, 6, 5, 4, 8, 9, 1, 7, 9, 10, 10])
kernel = np.array([-1, 0, 1])
n = len(input_values)

app_ui = ui.page_fluid(
    ui.tags.style("""
        .vertical-wrapper {
            display: flex;
            align-items: center;
            height: 250px;
        }

        .vertical-slider input[type=range] {
            writing-mode: bt-lr; /* fallback for older browsers */
            transform: rotate(270deg);
            width: 200px;
            height: 30px;
            margin-left: 10px;
        }

        .vertical-label {
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            margin-right: 10px;
        }
    """),

    ui.div(
        {"class": "vertical-wrapper"},
        ui.div("Kernel Position", {"class": "vertical-label"}),
        ui.div(
            ui.input_slider("position2", None, 0, 6, 0, step=1),
            {"class": "vertical-slider"}
        )
    ),

    ui.output_text("pos_output"),

    ui.h3("1D Convolution Visualizer"),
    ui.input_slider("position", "Kernel Position", 0, n - 3, 0),
    ui.output_ui("input_display"),
    ui.output_ui("kernel_display"),
    ui.output_ui("calc_result"),
)

def server(input, output, session):

    @output
    @render.ui
    def input_display():
        pos = input.position()
        tags = []
        for i, val in enumerate(input_values):
            bg = "#ffcccc" if pos <= i <= pos + 2 else "#ffffff"
            tags.append(
                ui.tags.div(
                    str(val),
                    style=f"border: 1px solid black; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; background-color: {bg};"
                )
            )
        return ui.tags.div({"style": "display: flex; gap: 4px;"}, *tags)

    @output
    @render.ui
    def kernel_display():
        tags = [
            ui.tags.div("Kernel:"),
            ui.tags.div(
                *[
                    ui.tags.div(
                        str(val),
                        style="border: 1px solid black; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;"
                    )
                    for val in kernel
                ],
                style="display: flex; gap: 4px; margin-top: 8px;"
            )
        ]
        return ui.tags.div(*tags)

    @output
    @render.ui
    def calc_result():
        pos = input.position()
        window = input_values[pos:pos+3]
        result = int(np.sum(window * kernel))
        detail = f"({window[0]}×{kernel[0]}) + ({window[1]}×{kernel[1]}) + ({window[2]}×{kernel[2]}) = {result}"
        return ui.tags.div(
            ui.tags.b("Calculation: "),
            detail,
            style="margin-top: 12px;"
        )

    @render.text
    def pos_output():
        return f"当前选择的位置是: {input.position()}"
    
app = App(app_ui, server)
