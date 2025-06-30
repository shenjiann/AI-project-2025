import numpy as np

def matrix_to_html(matrix: np.ndarray, highlight: list[tuple[int, int]] = None) -> str:
    """
    将矩阵转换为带左右中括号的 HTML 表格，并高亮特定元素。
    """
    if highlight is None:
        highlight = []

    html = '<div class="matrix-container"><table class="matrix">'
    for i in range(matrix.shape[0]):
        html += '<tr>'
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if (i, j) in highlight:
                html += f'<td class="highlight">{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</table></div>'
    return html