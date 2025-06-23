import numpy as np
import torch
import torch.nn.functional as F

kernel_choices = {
    'Random': '随机整数',
    'Smooth': '平均核',
    'Gaussian': '高斯核',
    'Sharpen': '锐化核',
    'Vedge': 'Sobel核(垂直边界)',
    'Hedge': 'Sobel核(水平边界)'
}

def generate_input(height:int, width:int, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 10, size=(height, width))

def generate_kernel(kernel:str, size:int, seed:int=42) -> np.ndarray:
    """
    生成卷积核
    """
    np.random.seed(seed)
    if kernel == 'Random':
        return np.random.randint(-5, 5, size=(size, size))
    elif kernel == 'Smooth':
        return np.ones((size, size)) / (size * size)
    elif kernel == 'Gaussian':
        def gaussian_2d(x, y, sigma=1.0):
            return np.exp(-(x**2 + y**2) / (2 * sigma**2))
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = gaussian_2d(xx, yy)
        kernel /= np.sum(kernel)
        return kernel
    elif kernel == 'Sharpen':
        kernel = np.zeros((size, size))
        kernel[size // 2, size // 2] = 2.0
        kernel += -1.0 / (size * size)
        return kernel
    elif kernel == 'Vedge':
        kernel = np.zeros((size, size))
        kernel[:, size // 2] = np.linspace(-1, 1, size)
        return kernel
    elif kernel == 'Hedge':
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = np.linspace(-1, 1, size)
        return kernel
    else:
        return np.eye(size)

def convolve(input:np.ndarray, kernel:np.ndarray, stride:int, padding:int) -> np.ndarray:
    """
    卷积计算
    """
    input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1,1,kH,kW)
    result = F.conv2d(input_tensor, kernel_tensor, stride=stride, padding=padding)
    return result.squeeze().detach().numpy()

# def mat_to_latex(kernel: np.ndarray, name: str = "K") -> str:
#     """
#     将 2D 矩阵转换为 LaTeX bmatrix 表达式
#     """
#     rows = []
#     for row in kernel:
#         row_str = " & ".join(str(int(val)) if val == int(val) else f"{val:.2f}" for val in row)
#         rows.append(row_str)
#     matrix_body = r" \\".join(rows)
#     return rf"\begin{{bmatrix}}{matrix_body}\end{{bmatrix}}"


def mat_to_latex(matrix: np.ndarray, wrap: bool = True) -> str:
    """
    2D矩阵转换为latex表达式
    """
    rows = [" & ".join(map(str, row)) for row in matrix]
    latex = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
    return f"$$ {latex} $$" if wrap else latex