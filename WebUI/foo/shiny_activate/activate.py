import numpy as np
import matplotlib.pyplot as plt
from shiny import App, render, ui, reactive
from pathlib import Path

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 图片路径
here = Path(__file__).parent

# 定义用户界面
app_ui = ui.page_fluid(
    # 关键修改1：图片容器（无间距）
    ui.div(
        ui.output_image("header_image"),
        style="margin:0; padding:0; height:120px; line-height:0;"
    ),
    
    # 关键修改2：自定义CSS样式
    ui.tags.style("""
        /* 全局重置 */
        body, html {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* 图片容器样式 */
        .shiny-image-output {
            margin: 0 !important;
            padding: 0 !important;
            display: block !important;
        }
        
        /* 布局容器 */
        .custom-container {
            margin-top: 0 !important;
            padding-top: 0 !important;
            overflow: hidden;
        }
        
        /* 左侧面板 */
        .left-panel {
            width: 30%;
            float: left;
            padding-right: 15px;
            box-sizing: border-box;
        }
        
        /* 右侧面板 */
        .right-panel {
            width: 70%;
            float: left;
            box-sizing: border-box;
        }
        
        /* 卡片样式调整 */
        .card {
            margin-top: 0 !important;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .left-panel, .right-panel {
                width: 100%;
                float: none;
                padding-right: 0;
            }
        }
    """),
    
    ui.panel_title("绘制激活函数"),
    
    # 关键修改3：使用自定义浮动布局替代layout_columns
    ui.div(
        # 左侧控制面板（30%宽度）
        ui.div(
            ui.card(
                ui.card_header("控制面板"),
                ui.card_body(
                    ui.input_select(
                        "activation",
                        "选择激活函数:",
                        {
                            "sigmoid": "Sigmoid",
                            "tanh": "Tanh",
                            "relu": "ReLU",
                            "leaky_relu": "Leaky ReLU"
                        },
                    ),
                    ui.input_slider(
                        "slope",
                        "斜率调整:",
                        min=0.01,
                        max=1,
                        value=0.1,
                        step=0.01
                    ),
                    ui.input_action_button("reset", "重置"),
                ),
                style="height: 100%;"
            ),
            class_="left-panel"
        ),
        
        # 右侧图像显示（70%宽度）
        ui.div(
            ui.card(
                ui.card_header("激活函数图像"),
                ui.output_plot("plot", width="100%", height="400px"),
                full_screen=True,
                style="height: 100%;"
            ),
            class_="right-panel"
        ),
        class_="custom-container"
    )
)

# 定义服务器逻辑
def server(input, output, session):
    # 关键修改4：图片渲染设置
    @render.image  
    def header_image():
        return {
            "src": here/"figs/threedep.png",
            "style": "width: 100%; height: 80px; display: block; margin: 0 !important; padding: 0 !important;"
        }

    @reactive.calc
    def activation_data():
        x = np.linspace(-10, 10, 400)
        activation = input.activation()
        slope = input.slope()

        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-x))
            dy = y * (1 - y)
            title = "Sigmoid 函数"
        elif activation == "tanh":
            y = np.tanh(x)
            dy = 1 - y**2
            title = "Tanh 函数"
        elif activation == "relu":
            y = np.maximum(0, x)
            dy = np.where(x >= 0, 1, 0)
            title = "ReLU 函数"
        elif activation == "leaky_relu":
            y = np.where(x >= 0, x, slope * x)
            dy = np.where(x >= 0, 1, slope)
            title = "Leaky ReLU 函数"
        else:
            y = x
            dy = np.ones_like(x)
            title = "线性函数"

        return x, y, dy, title

    @output
    @render.plot
    def plot():
        x, y, dy, title = activation_data()
        fig, ax = plt.subplots()
        ax.plot(x, y, label='函数')
        ax.plot(x, dy, label='导数', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x) 和 f'(x)")
        ax.legend()
        ax.grid(True)
        return fig

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_select("activation", selected="sigmoid")
        ui.update_slider("slope", value=0.1)

# 创建 Shiny 应用实例
app = App(app_ui, server)