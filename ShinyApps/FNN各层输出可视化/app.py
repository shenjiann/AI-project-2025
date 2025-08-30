from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
from utility import *

plt.rcParams["font.family"] = ["Times New Roman", "Songti SC"]

app_ui = ui.page_fluid(
    ui.include_css(APP_DIR/"www/styles.css"),
    ui.HTML((APP_DIR/ "www/mathjax_config.html").read_text(encoding="utf-8")),
    ui.output_image("threedep"),

    ui.panel_title("全连接神经网络各层输出可视化"),

    ui.layout_columns(
        ui.card(
            ui.card_header("选择MNIST图片"),
            ui.input_select(
                "digit", "选择数字类别", 
                choices=[str(i) for i in range(10)], 
                selected="0"),
            ui.input_select(
                "idx", 
                "选择样本(1-10)", 
                choices=[str(i) for i in range(1, 11)], 
                selected="1"),
            '当前选择图片：',
            ui.output_image("sel_img", width='100px', height='120px'),
        ),
        ui.card(
            ui.card_header("全连接层权重直方图"),
            ui.div(
                {"style": "display:flex; align-items:baseline; gap:12px;"},
                ui.span("选择层："),
                ui.input_radio_buttons(
                    "which_fc", None,
                    {"fc1": "隐藏层 1", 
                     "fc2": "隐藏层 2", 
                     "fc3": "输出层"},
                    selected="fc1", 
                    inline=True
                )
            ),
            ui.output_plot("weight_hist", height="300px"),
        ),
        col_widths=(4, 8),
    ),

    ui.accordion(  
        ui.accordion_panel(
            r"输入层 \( d^{[0]} = 784 \)", 
            ui.output_ui("x_dots")),  
        ui.accordion_panel(
            r"隐藏层 1 \( d^{[1]} = 128 \)", 
            ui.output_ui("h1_dots")),
        ui.accordion_panel(
            r"隐藏层 2 \( d^{[2]} = 64 \)", 
            ui.output_ui("h2_dots")),
        ui.accordion_panel(
            r"输出层 \( d^{[3]} = 10 \)", 
            ui.output_ui("y_dots"),
            ui.div( # 居中容器
                {"style": "text-align:center; margin-top:8px;"},
                ui.output_text("y_pred")
            )
        ),
        id="acc",  
        open=True,  
    ),
)
    
def server(input, output, session):
    # 顶部叠图
    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0;"
        }

    @render.image
    def sel_img():
        # 读取本地图片 {digit}_{k}.png
        digit = input.digit()
        k = int(input.idx()) - 1
        img_path = SAMPLES_DIR / f"{digit}_{k}.png"
        return {
            "src": img_path,
            "alt": f"MNIST {digit} 第 {k+1} 张",
            "style": "margin-top: 8px; width: 100px; height: 100px; align-items: center;"
        }
    
    @reactive.Calc
    def layer_vectors():
        x_vec = read_selected_image_to_vec(
            digit=input.digit(),
            idx=input.idx(),
        )        
        x, h1, h2, y = forward_pass(x_vec, *weights)
        return {"x": x, "h1": h1, "h2": h2, "y": y}


    @output
    @render.ui
    def x_dots():
        html = vector_to_html_dots(
            layer_vectors()['x'], 
            width_px=14, 
            gap_px=4, 
            pad_px=4,
            scroll=True)
        return ui.HTML(html)
    
    @output
    @render.ui
    def h1_dots():
        html = vector_to_html_dots(
            layer_vectors()['h1'], 
            width_px=14, 
            gap_px=4, 
            pad_px=4,
            scroll=True)
        return ui.HTML(html)
    
    @output
    @render.ui
    def h2_dots():
        html = vector_to_html_dots(
            layer_vectors()['h2'], 
            width_px=14, 
            gap_px=4, 
            pad_px=4,
            scroll=True)
        return ui.HTML(html)

    @output
    @render.ui
    def y_dots():
        html = vector_to_html_dots(
            layer_vectors()['y'], 
            width_px=14, 
            gap_px=4, 
            pad_px=4,
            scroll=True)
        return ui.HTML(html)
    
    @output
    @render.text
    def y_pred():
        y = layer_vectors()['y']
        pred = int(np.argmax(y))          # 取最大分量的类别
        return f"估计结果： {pred}"

    @output
    @render.plot
    def weight_hist():
        which = input.which_fc()
        w = pick_weights(which)
        return plot_weight_hist_simple(w, bins=60, xlim=(-6, 4))
    
app = App(app_ui, server)
