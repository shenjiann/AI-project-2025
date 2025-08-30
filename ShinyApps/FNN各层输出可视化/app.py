from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import io, math
from utility import *

app_ui = ui.page_fluid(
    ui.include_css(APP_DIR/"www/styles.css"),
    ui.output_image("threedep"),

    ui.panel_title("全连接神经网络各层输出可视化"),

    ui.accordion(  
            ui.accordion_panel(
                "选择MNIST图片",
            ui.layout_column_wrap(
                    1/2, # 指定每行放置2个元素
                    ui.input_select(
                        "digit", "选择数字类别", 
                        choices=[str(i) for i in range(10)], 
                        selected="0"),
                    ui.input_select(
                        "idx", 
                        "选择样本(1-10)", 
                        choices=[str(i) for i in range(1, 11)], 
                        selected="1"),
                ),
                ui.output_image("sel_img")
            ),
            ui.accordion_panel(
                "输入层", 
                ui.output_ui("x_dots")),  
            ui.accordion_panel(
                "隐藏层1", 
                ui.output_ui("h1_dots"),
                ui.output_plot("fc1_hist")),  
            ui.accordion_panel(
                "隐藏层2", 
                ui.output_ui("h2_dots"),
                ui.output_plot("fc2_hist")),  
            ui.accordion_panel(
                "输出层", 
                ui.output_ui("y_dots"),
                ui.output_plot("fc3_hist")),  
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
            "style": "margin-top: 8px;"
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
    @render.plot
    def fc1_hist():
        try:
            fc1_w = weights[0]
            return plot_weight_hist(fc1_w, title="FC1 Weights")
        except Exception:
            fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
            ax.text(0.5, 0.5, "FC1 weights not found", ha="center", va="center")
            ax.axis("off")
            return fig

    @output
    @render.plot
    def fc2_hist():
        try:
            fc2_w = weights[2]
            return plot_weight_hist(fc2_w, title="FC2 Weights")
        except Exception:
            fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
            ax.text(0.5, 0.5, "FC2 weights not found", ha="center", va="center")
            ax.axis("off")
            return fig

    @output
    @render.plot
    def fc3_hist():
        try:
            fc3_w = weights[4]
            return plot_weight_hist(fc3_w, title="FC3 / Output Weights")
        except Exception:
            fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
            ax.text(0.5, 0.5, "FC3 weights not found", ha="center", va="center")
            ax.axis("off")
            return fig
app = App(app_ui, server)
