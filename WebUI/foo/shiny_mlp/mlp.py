import numpy as np
from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

here = Path(__file__).parent

def mat2latex(matrix, precision=2):
    """矩阵转LaTeX函数，支持上标标注"""
    if matrix.ndim == 1:
        elements = ' & '.join([f'{val:.{precision}f}' for val in matrix])
        return f'{elements}'
    elif matrix.ndim == 2:
        rows = []
        for row in matrix:
            elements = ' & '.join([f'{val:.{precision}f}' for val in row])
            rows.append(elements)
        return ' \\\\ '.join(rows)
    else:
        return ''

app_ui = ui.page_fluid(
    # 顶部图片
    ui.div(
        ui.output_image("header_image"),
        style="margin:0; padding:0; height:120px; line-height:0;"
    ),
    
    # 标题面板
    ui.panel_title("多个隐藏层NLP参数维度"),
    
    # 自定义CSS样式
    ui.tags.style("""
        body, html {
            margin: 0 !important;
            padding: 0 !important;
        }
        .shiny-image-output {
            margin: 0 !important;
            padding: 0 !important;
        }
        .custom-container {
            margin-top: 0 !important;
            padding-top: 0 !important;
            overflow: hidden;
        }
        .left-panel {
            width: 30%;
            float: left;
            padding-right: 15px;
            box-sizing: border-box;
        }
        .right-panel {
            width: 70%;
            float: left;
            box-sizing: border-box;
        }
        .dimension-slider {
            margin-bottom: 15px;
        }
        .fixed-dimension {
            color: #666;
            font-style: italic;
            margin: 10px 0;
        }
        @media (max-width: 768px) {
            .left-panel, .right-panel {
                width: 100%;
                float: none;
                padding-right: 0;
            }
        }
    """),
    
    # MathJax配置
    ui.HTML(r"""
    <script>
    MathJax = {
      tex: {
        packages: {'[+]': ['ams']},
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
      },
      loader: {load: ['[tex]/ams']},
      startup: {
        typeset: false
      }
    };
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    """),
    
    # 主布局
    ui.div(
        # 左侧控制面板
        ui.div(
            ui.card(
                ui.card_header("控制面板"),
                ui.card_body(
                    ui.input_slider("layers", "网络层数L:", min=1, max=5, value=2, step=1),
                    ui.input_slider("input_dim", "输入维度d[0]:", min=1, max=5, value=3, step=1),
                    *[ui.input_slider(
                        f"hidden_dim_{i}", 
                        f"隐藏层d[{i}]:", 
                        min=1, 
                        max=5, 
                        value=2,
                        step=1
                      ) for i in range(1, 5)],
                    ui.div(
                        "输出层固定: d[L] = 1",
                        class_="fixed-dimension"
                    ),
                    ui.input_action_button("refresh", "刷新参数", style="width:100%; margin-top:15px;"),
                ),
                style="height:100%;"
            ),
            class_="left-panel"
        ),
        
        # 右侧矩阵显示
        ui.div(
            ui.card(
                ui.card_header("网络参数矩阵"),
                ui.output_ui("matrix_output"),
                style="height:100%;"
            ),
            class_="right-panel"
        ),
        class_="custom-container"
    )
)

def server(input, output, session):
    # 图片渲染
    @render.image  
    def header_image():
        return {
            "src": here/"figs/threedep.png",
            "style": "width:100%; height:80px; display:block; margin:0; padding:0;"
        }
    
    # 动态禁用不需要的隐藏层滑块
    @reactive.effect
    def _():
        l = input.layers()
        for i in range(1, 5):
            if i < l:
                current_value = input[f"hidden_dim_{i}"]()
                ui.update_slider(
                    f"hidden_dim_{i}", 
                    label=f"隐藏层d[{i}]:",
                    min=1,
                    max=5,
                    value=2 if current_value == 0 else current_value
                )
            else:
                ui.update_slider(
                    f"hidden_dim_{i}", 
                    label=None,
                    value=0,
                    min=0,
                    max=0
                )

    # 参数生成逻辑
    weights = reactive.Value(None)
    biases = reactive.Value(None)
    dimensions = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.refresh)
    def update_parameters():
        l = input.layers()
        dims = [input.input_dim()]  # d[0]
        
        # 收集各层维度 (d[1]到d[L])
        for i in range(1, l+1):
            if i == l:  # 输出层固定为1
                dims.append(1)
            else:
                dims.append(input[f"hidden_dim_{i}"]())
        
        # 生成参数（修正维度：W[l] ∈ R^{d[l] × d[l-1]}）
        new_weights = []
        new_biases = []
        
        for i in range(l):
            w = np.random.randn(dims[i+1], dims[i]).round(2)  # 修正为 (d[l], d[l-1])
            b = np.random.randn(dims[i+1]).round(2)
            new_weights.append(w)
            new_biases.append(b)
        
        dimensions.set(dims)
        weights.set(new_weights)
        biases.set(new_biases)

    # 矩阵显示
    @output
    @render.ui
    def matrix_output():
        if weights.get() is None:
            return ui.HTML("""
                <div style="text-align: center; margin-top: 20px;">
                    <p>请点击'刷新参数'生成网络参数</p>
                    <p class="fixed-dimension">输出层固定: d[L] = 1</p>
                </div>
            """)
        
        dims = dimensions.get()
        weights_list = weights.get()
        biases_list = biases.get()
        
        result = []
        
        # 显示输入维度
        result.append(rf"\[ \text{{输入维度}}: d^{{[0]}} = {dims[0]} \]")
        
        for i in range(len(weights_list)):
            # 当前层参数
            w = weights_list[i]
            b = biases_list[i]
            
            # 生成LaTeX矩阵
            w_latex = mat2latex(w)
            b_latex = mat2latex(b)
            
            # 构建表达式（标注维度）
            layer_label = f"{i+1}" if i+1 < len(dims)-1 else "L"
            expr = rf"""
            \[
            W^{{[{layer_label}]}} \in \mathbb{{R}}^{{{dims[i+1]} \times {dims[i]}}}: 
            \begin{{bmatrix}}
            {w_latex}
            \end{{bmatrix}}
            \quad
            b^{{[{layer_label}]}} \in \mathbb{{R}}^{{{dims[i+1]}}}: 
            \begin{{bmatrix}}
            {b_latex}
            \end{{bmatrix}}
            \]
            """
            result.append(expr)
            
            # 添加层间箭头
            if i < len(weights_list) - 1:
                result.append(rf"\[ \downarrow \text{{层 {i+1} 到层 {i+2}}} \]")
            else:
                result.append(rf"\[ \text{{输出层固定}}: d^{{[L]}} = 1 \]")
        
        return ui.HTML(f"""
            <div style="text-align: center; margin-top: 10px;">
                {'<hr style="margin: 20px 0;">'.join(result)}
            </div>
            <script>
                if (window.MathJax) {{
                    MathJax.typesetPromise();
                }}
            </script>
        """)

app = App(app_ui, server)