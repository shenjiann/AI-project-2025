import numpy as np
import torch
import torch.nn.functional as F

pool_modes = {
    'avg': '平均池化',
    'max': '最大池化',
}

def generate_input(height:int, width:int, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 10, size=(height, width))

def pool(input_array: np.ndarray, kernel_size: int, stride: int, padding: int, mode: str = "max") -> np.ndarray:
    x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if mode == "max":
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    elif mode == 'avg':
        y = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Expected 'max' or 'avg'")
    return y.squeeze().detach().numpy()

def mat2latex(matrix: np.ndarray, wrap: bool = True) -> str:
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