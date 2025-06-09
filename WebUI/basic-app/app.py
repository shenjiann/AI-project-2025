from shiny import reactive 
from shiny.express import input, render, ui
from PIL import Image
import numpy as np
import tempfile

ui.input_file('f', 'upload a file')

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

@render.text
def image_matrix():
    arr = img_array()
    # 格式化输出矩阵，每个像素值占4个字符宽度
    formatted_rows = [" ".join(f"{val:3}" for val in row) for row in arr]
    return "\n".join(formatted_rows)


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

ui.input_slider('padding', 'padding', 0, 10, 1)
ui.input_slider('stride', 'stride', 0, 10, 1)