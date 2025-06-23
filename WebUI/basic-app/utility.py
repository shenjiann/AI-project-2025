import numpy as np
import torch
import torch.nn.functional as F

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

import numpy as np
from math import comb # 需要导入 comb 来生成更高级的平滑核

def generate_kernel(kernel:str, size:int, seed:int=42) -> np.ndarray:
    """
    生成卷积核 (修正了Sobel核的实现)
    """
    np.random.seed(seed)

    # 确保尺寸为奇数，便于中心对称
    if size % 2 == 0:
        size += 1

    if kernel == 'Random':
        return np.random.randint(-5, 5, size=(size, size))
    elif kernel == 'Smooth':
        return np.ones((size, size)) / (size * size)
    elif kernel == 'Gaussian':
        # 高斯核的实现保持不变
        sigma = max(1.0, size / 5.0) # 让sigma随尺寸变化
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    elif kernel == 'Sharpen':
        # 锐化核的实现保持不变
        center = size // 2
        kernel = -np.ones((size, size)) / (size * size)
        kernel[center, center] = 2.0 - 1.0 / (size*size)
        return kernel

    # --- Sobel 核的修正部分 ---
    elif kernel == 'SobelVert' or kernel == 'SobelHori':
        # 对于 size=3，使用经典的Sobel向量 [1, 2, 1]
        # 对于其他尺寸，使用更通用的平滑向量（如二项式系数或简单的ones）
        if size == 3:
            smooth_vec = np.array([1, 2, 1])
            deriv_vec = np.array([-1, 0, 1])
        else:
            # 使用二项式系数作为更通用的平滑核（类似高斯）
            smooth_vec = np.array([comb(size - 1, i) for i in range(size)])
            # 使用简单的中心差分作为导数核
            deriv_vec = np.zeros(size)
            deriv_vec[0] = -1
            deriv_vec[-1] = 1

        if kernel == 'SobelVert':
            # 垂直梯度 (用于检测水平边缘)
            # 通过 垂直的微分向量 和 水平的平滑向量 的外积得到
            return np.outer(deriv_vec, smooth_vec)
        else: # SobelHori
            # 水平梯度 (用于检测垂直边缘)
            # 通过 垂直的平滑向量 和 水平的微分向量 的外积得到
            return np.outer(smooth_vec, deriv_vec)


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


import numpy as np

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