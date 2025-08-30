from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import io, math
from utility import *

app_ui = ui.page_fluid(
    ui.include_css(APP_DIR/"www/styles.css"),
    ui.output_image("threedep"),

    ui.panel_title("全连接神经网络各层输出可视化"),
    ui.layout_columns(
        ui.card(
            ui.card_header('选择MNIST图片'),
            ui.input_select(
                "digit", "选择数字类别", 
                choices=[str(i) for i in range(10)], 
                selected="0"),
            ui.input_select(
                "idx", 
                "选择样本(1-10)", 
                choices=[str(i) for i in range(1, 11)], 
                selected="1"),
            '当前选择的图片：',
            ui.output_image("sel_img")
        ),
        ui.card(
            ui.card_header('aaa'),
            ui.output_ui("output_vec_view"),
            ui.output_ui("h1_dots"),
            ui.output_ui("y_dots")
        )
    )
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
    def h1_dots():
        html = vector_to_html_dots(
            layer_vectors()['h1'], width_px=14, gap_px=4, pad_px=4)
        return ui.HTML(html)

    @output
    @render.ui
    def y_dots():
        html = vector_to_html_dots(
            layer_vectors()['y'], width_px=14, gap_px=4, pad_px=4)
        return ui.HTML(html)
    

app = App(app_ui, server)
