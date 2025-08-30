import numpy as np
from shiny import App, render, ui, reactive
from pathlib import Path
from utility import *
import torch
import matplotlib.pyplot as plt

# 设置图片字体
plt.rcParams["font.family"] = ["Times New Roman", "Songti SC"]

app_ui = ui.page_fluid(
    ui.include_css(APP_DIR/"www/styles.css"),
    ui.HTML((APP_DIR/ "www/mathjax_config.html").read_text(encoding="utf-8")),
    ui.output_image("threedep"),
    ui.panel_title("卷积神经网络各层输出可视化"),
    ui.layout_columns(
        ui.card(
            ui.card_header("选择CIFAR-10图片"),
            ui.input_select(
                "cls", "选择类别", 
                choices=[
                    "airplane", "automobile", "bird", "cat", "deer", 
                    "dog", "frog", "horse", "ship", "truck"
                ], 
                selected="airplane"),
            ui.input_select(
                "idx", 
                "选择样本(1-10)",
                choices=[str(i) for i in range(1, 11)], 
                selected="1"),
            '当前选择图片：',
            ui.output_image("sel_img", width='100px', height='100px'),
        ),
        ui.card(
            ui.card_header("卷积层权重直方图"),
            ui.div(
                {"style": "display:flex; align-items:baseline; gap:12px;"},
                ui.span("选择层："),
                ui.input_radio_buttons(
                    "which_conv", None,
                    {"conv1": "卷积层 1", 
                     "conv2": "卷积层 2", 
                     "conv3": "卷积层 3",
                     "conv4": "卷积层 4", 
                     "conv5": "卷积层 5",},
                    selected="conv1", 
                    inline=True
                )
            ),
            ui.output_plot("weight_hist", height="300px"),
        ),
        col_widths=(4, 8),
    ),
    ui.accordion(
        ui.accordion_panel(
            r"卷积层 1 特征图",
            ui.output_plot("feat_conv1", height="400px"),
        ),
        ui.accordion_panel(
            r"卷积层 2 特征图",
            ui.output_plot("feat_conv2", height="400px"),
        ),
        ui.accordion_panel(
            r"卷积层 3 特征图",
            ui.output_plot("feat_conv3", height="400px"),
        ),
        ui.accordion_panel(
            r"卷积层 4 特征图",
            ui.output_plot("feat_conv4", height="400px"),
        ),
        ui.accordion_panel(
            r"卷积层 5 特征图",
            ui.output_plot("feat_conv5", height="400px"),
        ),
        id="acc",
        open=True,
    ),
    ui.row(
        ui.column(4, 
            ui.input_select(
                "pick_method", "代表性通道选择策略",
                choices={
                    "energy": "能量（默认）",
                    "variance": "空间方差",
                    "max": "最大响应",
                    "sparse": "稀疏激活优先"
                }, selected="energy"
            )
        ),
        ui.column(4,
            ui.input_slider("topk", "每层展示通道数", min=4, max=16, value=8, step=1)
        ),
    ),
)


def server(input, output, session):
    # --- 图片 ---
    @render.image
    def threedep():
        return {
            "src": Path(__file__).parent/"www/threedep.png",
            "style": "position: absolute; top: 0; left: 0;"
        }
    
    @render.image
    def sel_img():
        cls = input.cls()
        k = int(input.idx()) - 1
        img_path = SAMPLES_DIR / f"{cls}_{k}.png"
        return {
            "src": img_path,
            "width": "100px",
            "height": "100px",
            "style": "border: 1px solid #ccc; border-radius: 4px; margin-top: 8px;"
        }
    
    @render.plot
    def weight_hist():
        which = input.which_conv()
        w = pick_conv_weights(which)
        return plot_weight_hist(w, bins=100)
    
    @reactive.Calc
    def _grids():
        cls = input.cls()
        k = int(input.idx()) - 1
        method = input.pick_method()
        topk = int(input.topk())
        x = load_sample_tensor(cls, k)
        grids = build_layer_grids(model, x, topk_per_layer=topk, method=method)
        return grids

    # --- 每层输出 ---
    @render.plot
    def feat_conv1():
        g = _grids()
        fig, meta = g["conv1"]
        return fig

    @render.plot
    def feat_conv2():
        g = _grids()
        fig, meta = g["conv2"]
        return fig

    @render.plot
    def feat_conv3():
        g = _grids()
        fig, meta = g["conv3"]
        return fig

    @render.plot
    def feat_conv4():
        g = _grids()
        fig, meta = g["conv4"]
        return fig

    @render.plot
    def feat_conv5():
        g = _grids()
        fig, meta = g["conv5"]
        return fig

app = App(app_ui, server)
