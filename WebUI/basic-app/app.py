from shiny import reactive 
from shiny.express import input, render, ui
from PIL import Image
import numpy as np
import tempfile


# ui.page_opts(fillable=True)

with ui.layout_sidebar():
    with ui.sidebar(bg='#f8f8f8'):
        ui.input_file('f', 'select a file')

    @reactive.calc
    def parsed_file():
        f_path = input.f()[0]['datapath']
        return f_path

    @reactive.calc
    def img_array():
        img = Image.open(parsed_file()).convert('L')
        return np.array(img)


    @render.image
    def image():
        img = Image.fromarray(img_array())
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(tmp.name)
        img = {'src': tmp.name, 'width': '100px'}
        return img




with ui.layout_sidebar():
    with ui.sidebar(bg='#f8f8f8'):
        ui.input_select(
            'conv_kernel_select',
            'select a convolution kernel',
            {
                'Gaussian': 'Gaussian',
                'Sharpen': 'Sharpen',
                'vertical edge': 'vertical edge',
                'horizontal edge': 'horizontal edge',
            }
        )
        ui.input_slider('size', 'size', 1, 10, 1)
        ui.input_slider('padding', 'padding', 0, 10, 0)
        ui.input_slider('stride', 'stride', 1, 10, 1)



