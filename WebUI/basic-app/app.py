from shiny import reactive, req
from shiny.express import input, render, ui
from PIL import Image
import numpy as np
import tempfile
import torch
import torch.nn.functional as F

kernel_chioces = {
    'Gaussian': 'Gaussian',
    'Sharpen': 'Sharpen',
    'vertical edge': 'vertical edge',
    'horizontal edge': 'horizontal edge',
}

with ui.sidebar(bg='#f8f8f8'):
    ui.input_file('f', 'select a file')
    ui.input_select(
        'kernel',
        'select a convolution kernel',
        kernel_chioces
        )
    ui.input_slider('size', 'size', 1, 10, 1)
    ui.input_slider('padding', 'padding', 0, 10, 0)
    ui.input_slider('stride', 'stride', 1, 10, 1)


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

    if kernel_type == 'Gaussian':
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



@render.image
def image():
    img = Image.fromarray(src_image())
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name)
    img = {'src': tmp.name, 'width': '300px'}
    return img

@render.image
def convolved_image():
    img = Image.fromarray(convolve_img())
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name)
    return {'src': tmp.name, 'width': '300px'}