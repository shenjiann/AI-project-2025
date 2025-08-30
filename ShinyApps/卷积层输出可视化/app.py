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
            ui.layout_columns(
                ui.input_selectize(
                    "ch_conv1", "选择通道（多选）",
                    choices=[], selected=[], multiple=True
                ),
                ui.input_numeric("max_cols1", "每行显示列数", 6, min=1, max=16),
            ),
            ui.output_plot("fm_conv1", height="420px"),
        ),
        ui.accordion_panel(
            r"卷积层 2 特征图",
            ui.layout_columns(
                ui.input_selectize(
                    "ch_conv2", "选择通道（多选）",
                    choices=[], selected=[], multiple=True
                ),
                ui.input_numeric("max_cols2", "每行显示列数", 6, min=1, max=16),
            ),
            ui.output_plot("fm_conv2", height="420px"),
        ),
        ui.accordion_panel(
            r"卷积层 3 特征图",
            ui.layout_columns(
                ui.input_selectize(
                    "ch_conv3", "选择通道（多选）",
                    choices=[], selected=[], multiple=True
                ),
                ui.input_numeric("max_cols3", "每行显示列数", 6, min=1, max=16),
            ),
            ui.output_plot("fm_conv3", height="420px"),
        ),
        ui.accordion_panel(
            r"卷积层 4 特征图",
            ui.layout_columns(
                ui.input_selectize(
                    "ch_conv4", "选择通道（多选）",
                    choices=[], selected=[], multiple=True
                ),
                ui.input_numeric("max_cols4", "每行显示列数", 6, min=1, max=16),
            ),
            ui.output_plot("fm_conv4", height="420px"),
        ),
        ui.accordion_panel(
            r"卷积层 5 特征图",
            ui.layout_columns(
                ui.input_selectize(
                    "ch_conv5", "选择通道（多选）",
                    choices=[], selected=[], multiple=True
                ),
                ui.input_numeric("max_cols5", "每行显示列数", 6, min=1, max=16),
            ),
            ui.output_plot("fm_conv5", height="420px"),
        ),
        id="acc",
        open=True,
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

    # —— 工具：根据权重形状动态设置多选通道列表
    def _update_channel_selector(id_select: str, which: str, n_default: int = 6):
        """把通道选择器的 choices 更新为 [0..C-1]，默认选前 n_default 个"""
        w = pick_conv_weights(which)  # (C_out, C_in, k, k)
        C_out = int(w.shape[0])
        choices = [str(i) for i in range(C_out)]
        selected = [str(i) for i in range(min(n_default, C_out))]
        ui.update_selectize(
            id_select,
            choices=choices,
            selected=selected,
            session=session,   # 关键：必须用关键字传递
        )

    # —— 首次进入页面/或图片选择变化时，刷新各层通道选择器
    @reactive.effect
    def _init_channel_choices():
        # 任意触发源；这里用类别/样本变化做一次同步
        _ = input.cls()
        _ = input.idx()
        _update_channel_selector("ch_conv1", "conv1")
        _update_channel_selector("ch_conv2", "conv2")
        _update_channel_selector("ch_conv3", "conv3")
        _update_channel_selector("ch_conv4", "conv4")
        _update_channel_selector("ch_conv5", "conv5")

    # —— 通用绘图：把若干通道的特征图做成网格
    def _plot_feature_maps_grid(fmaps: np.ndarray, channels: list[int], max_cols: int = 6, title: str = ""):
        """
        fmaps: [C, H, W] 的 numpy 数组（数值已归一化到 0~1 更佳）
        channels: 需要显示的通道编号（int 列表）
        max_cols: 每行最多列数
        """
        if fmaps.ndim != 3:
            raise ValueError("feature maps should have shape [C, H, W].")
        if len(channels) == 0:
            # 如果没选，就默认显示前 6 个
            channels = list(range(min(6, fmaps.shape[0])))

        cols = max(1, int(max_cols))
        rows = (len(channels) + cols - 1) // cols

        # 统一对每张通道图做 min-max 归一化，提升可视对比
        def _normalize(x):
            x = x.astype(np.float32)
            mn, mx = float(x.min()), float(x.max())
            return (x - mn) / (mx - mn + 1e-8)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), dpi=150)
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = np.array([axs])
        elif cols == 1:
            axs = np.array([[ax] for ax in axs])

        for i, ch in enumerate(channels):
            r, c = divmod(i, cols)
            ax = axs[r, c]
            im = _normalize(fmaps[ch])
            ax.imshow(im, cmap="gray", interpolation="nearest")
            ax.set_title(f"ch {ch}", fontsize=9)
            ax.axis("off")

        # 把多出来的空格子关掉
        for i in range(len(channels), rows * cols):
            r, c = divmod(i, cols)
            axs[r, c].axis("off")

        if title:
            fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig

    # —— 获取当前选中图片的路径
    def _current_img_path():
        cls = input.cls()
        k = int(input.idx()) - 1
        return SAMPLES_DIR / f"{cls}_{k}.png"

    # 下列五个输出依赖一个工具函数：get_feature_maps(which, img_path) -> np.ndarray[C,H,W]
    # 若你还没实现，请把我文末的 utility.py 参考实现粘进去即可。

    @output
    @render.plot
    def fm_conv1():
        img_path = _current_img_path()
        fmaps = get_feature_maps("conv1", img_path)
        chs = [int(x) for x in (input.ch_conv1() or [])]
        return _plot_feature_maps_grid(fmaps, chs, int(input.max_cols1()), "Conv1 Feature Maps")

    @output
    @render.plot
    def fm_conv2():
        img_path = _current_img_path()
        fmaps = get_feature_maps("conv2", img_path)
        chs = [int(x) for x in (input.ch_conv2() or [])]
        return _plot_feature_maps_grid(fmaps, chs, int(input.max_cols2()), "Conv2 Feature Maps")

    @output
    @render.plot
    def fm_conv3():
        img_path = _current_img_path()
        fmaps = get_feature_maps("conv3", img_path)
        chs = [int(x) for x in (input.ch_conv3() or [])]
        return _plot_feature_maps_grid(fmaps, chs, int(input.max_cols3()), "Conv3 Feature Maps")

    @output
    @render.plot
    def fm_conv4():
        img_path = _current_img_path()
        fmaps = get_feature_maps("conv4", img_path)
        chs = [int(x) for x in (input.ch_conv4() or [])]
        return _plot_feature_maps_grid(fmaps, chs, int(input.max_cols4()), "Conv4 Feature Maps")

    @output
    @render.plot
    def fm_conv5():
        img_path = _current_img_path()
        fmaps = get_feature_maps("conv5", img_path)
        chs = [int(x) for x in (input.ch_conv5() or [])]
        return _plot_feature_maps_grid(fmaps, chs, int(input.max_cols5()), "Conv5 Feature Maps")


app = App(app_ui, server)
