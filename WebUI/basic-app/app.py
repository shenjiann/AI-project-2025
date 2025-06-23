from shiny import reactive, req
from shiny.express import input, render, ui
from PIL import Image
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
from pathlib import Path

ui.HTML("""
<script type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
""")

kernel_chioces = {
    'Smooth': '平均核',
    'Gaussian': '高斯核',
    'Sharpen': '锐化核',
    'vertical edge': 'Sobel核(垂直边界)',
    'horizontal edge': 'Sobel核(水平边界)'
}

@render.image  
def threedep():
    here = Path(__file__).parent
    return {"src": here / "figs/threedep.png", 'width': '100%'}


@render.ui
def matrix_latex():
    A = np.array([[1, 2], [3, 4]])
    # 将矩阵格式化成 LaTeX bmatrix 表达式
    matrix_str = "\\begin{bmatrix}" + " \\\\ ".join(
        [" & ".join(map(str, row)) for row in A]
    ) + "\\end{bmatrix}"

    return ui.HTML(f"$$ {matrix_str} $$")

with ui.card():  
    ui.card_header("Card with sidebar")

    with ui.layout_sidebar():  
        with ui.sidebar(bg="#f8f8f8"):  
            ui.input_file('f', '选择图片', accept='image/*')
            ui.input_select(
                'kernel',
                '选择卷积核',
                kernel_chioces
                )
            ui.input_slider('size', 'f', 1, 10, 1)
            ui.input_slider('padding', 'p', 0, 10, 0)
            ui.input_slider('stride', 's', 1, 10, 1)

        "Card content"  
        @render.text
        def text1():
            return f'Before Convolution'

        @render.image
        def image():
            img = Image.fromarray(src_image())
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name)
            img = {'src': tmp.name, 'width': '300px'}
            return img

        @render.text
        def text2():
            return f'After Convolution'

        @render.image
        def convolved_image():
            img = Image.fromarray(convolve_img())
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name)
            return {'src': tmp.name, 'width': '300px'}

@reactive.calc
def parsed_file():
    f_path = req(input.f())[0]['datapath']
    return f_path

@reactive.calc
def src_image():
    image = parsed_file()
    image = Image.open(image).convert('L')
    array = np.array(image)
    return array

@reactive.calc
def generate_kernel():
    kernel_type = input.kernel()
    size = input.size()

    if kernel_type == 'Smooth':
        return np.ones((size, size)) / (size * size)

    elif kernel_type == 'Gaussian':
        def gaussian_2d(x, y, sigma=1.0):
            return np.exp(-(x**2 + y**2) / (2 * sigma**2))

        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = gaussian_2d(xx, yy)
        kernel /= np.sum(kernel)
        return kernel

    elif kernel_type == 'Sharpen':
        kernel = np.zeros((size, size))
        kernel[size // 2, size // 2] = 2.0
        kernel += -1.0 / (size * size)
        return kernel

    elif kernel_type == 'vertical edge':
        kernel = np.zeros((size, size))
        kernel[:, size // 2] = np.linspace(-1, 1, size)
        return kernel

    elif kernel_type == 'horizontal edge':
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = np.linspace(-1, 1, size)
        return kernel

    else:
        return np.eye(size)  # fallback

@reactive.calc
def convolve_img():
    array = src_image()
    kernel = generate_kernel()

    img_tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
    padding = input.padding()
    stride = input.stride()

    output = F.conv2d(img_tensor, kernel_tensor, padding=padding, stride=stride)
    output_array = output.squeeze().numpy()
    output_array = (output_array - output_array.min()) * (255.0 / (output_array.max() - output_array.min()))
    output_array = output_array.astype(np.uint8)
    
    return output_array


