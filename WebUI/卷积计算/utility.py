import numpy as np
import torch
import torch.nn.functional as F
from math import comb

kernel_choices = {
    'Random': '随机整数',
    'Smooth': '平均核',
    'Gaussian': '高斯核',
    'Sharpen': '锐化核',
    'SobelVert': 'Sobel核(垂直边界)',
    'SobelHori': 'Sobel核(水平边界)'
}

def generate_input(height:int, width:int, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 10, size=(height, width))

def generate_kernel(kernel: str, size: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)

    if kernel == 'Random':
        return np.random.randint(-5, 5, size=(size, size))

    elif kernel == 'Smooth':
        return np.ones((size, size)) / (size * size)

    elif kernel == 'Gaussian':
        sigma = max(1.0, size / 5.0)
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    elif kernel == 'Sharpen':
        kernel = -np.ones((size, size)) / (size * size)
        center = (size - 1) // 2 if size % 2 == 1 else (size // 2 - 1)
        kernel[center, center] = 2.0 - 1.0 / (size * size)
        return kernel

    elif kernel == 'SobelVert' or kernel == 'SobelHori':
        # 平滑向量：采用二项式系数（近似高斯）
        smooth_vec = np.array([comb(size - 1, i) for i in range(size)])
        
        # 导数向量（中心差分）
        deriv_vec = np.zeros(size)
        deriv_vec[0] = -1
        deriv_vec[-1] = 1

        if kernel == 'SobelHori':
            return np.outer(deriv_vec, smooth_vec)
        else:  # SobelHori
            return np.outer(smooth_vec, deriv_vec)

def convolve(input:np.ndarray, kernel:np.ndarray, stride:int, padding:int) -> np.ndarray:
    """
    卷积计算
    """
    input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1,1,kH,kW)
    result = F.conv2d(input_tensor, kernel_tensor, stride=stride, padding=padding)
    return result.squeeze().detach().numpy()

def mat_to_latex(matrix: np.ndarray, wrap: bool = True) -> str:
    """
    2D矩阵转换为latex表达式
    """
    def format_element(element):
        if element == int(element):
            return str(int(element))
        else:
            return f"{element:.2f}" # 非整数元素保留2位小数

    rows = [" & ".join(map(format_element, row)) for row in matrix]
    latex = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
    return f"$$ {latex} $$" if wrap else latex