import numpy as np

def matrix_to_html(matrix: np.ndarray, highlight: list[tuple[int, int]] = None) -> str:
    """
    将矩阵转换为 HTML 网格，并根据给定坐标高亮特定元素。
    支持动态调整列数。
    """
    if highlight is None:
        highlight = []

    n_cols = matrix.shape[1]
    grid_style = f'display: grid; grid-template-columns: repeat({n_cols}, 40px); grid-gap: 4px; margin-top: 20px;'

    html = f'<div class="matrix-grid" style="{grid_style}">'
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if (i, j) in highlight:
                cell_class = "highlight"
            else:
                cell_class = "cell"
            html += f'<div class="{cell_class}">{val}</div>'
    html += '</div>'
    return html