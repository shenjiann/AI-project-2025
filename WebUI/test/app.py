from shiny import App, ui, render, reactive
import numpy as np
import torch

# 卷积参数
Z = torch.randint(-5, 6, (1, 1, 4, 4), dtype=torch.float32)
W = torch.randint(-2, 3, (1, 1, 2, 2), dtype=torch.float32)
dZ0 = torch.randint(-1, 2, (1, 1, 3, 3), dtype=torch.float32)
dZ = torch.zeros_like(Z)

stride = 1
padding = 0

# 当前步数和自动播放状态
current_step = reactive.Value(0)
auto_play = reactive.Value(False)

def matrix_to_html(matrix: torch.Tensor, highlight=None, prefix="") -> str:
    if highlight is None:
        highlight = []
    array = matrix[0, 0].detach().numpy()
    html = '<table style="border-collapse: collapse;">'
    for i in range(array.shape[0]):
        html += "<tr>"
        for j in range(array.shape[1]):
            val = int(array[i, j])
            style = "border: 1px solid black; width: 30px; height: 30px; text-align: center;"
            if (i, j) in highlight:
                style += " background-color: yellow;"
            html += f'<td style="{style}">{val}</td>'
        html += "</tr>"
    html += "</table>"
    return f'<div style="margin: 10px;">{prefix}{html}</div>'

# UI
app_ui = ui.page_fluid(
    ui.h3("反向传播中 dZ 的更新过程"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_action_button("next", "下一步"),
            ui.input_action_button("start", "自动播放 ▶️"),
            ui.input_action_button("stop", "停止 ⏸")
        ),
        ui.output_ui("dz0_display"),
        ui.output_ui("dz_display")
    )
)

# Server
def server(input, output, session):
    # 手动推进一步
    @reactive.effect
    @reactive.event(input.next)
    def advance_once():
        advance_step()

    # 自动播放控制
    @reactive.effect
    @reactive.event(input.start)
    def _start():
        auto_play.set(True)

    @reactive.effect
    @reactive.event(input.stop)
    def _stop():
        auto_play.set(False)

    # 自动播放动画机制
    @reactive.effect
    def auto_loop():
        if auto_play() and current_step() < 2:
            reactive.invalidate_later(2000)  # 提前设置等待 2 秒
            advance_step()
        elif current_step() >= 9:
            auto_play.set(False)  # 自动停止
            
    def advance_step():
        k = current_step()
        if k < 9:
            i = k // 3
            j = k % 3
            grad = dZ0[0, 0, i, j]
            for m in range(2):
                for n in range(2):
                    dZ[0, 0, i + m, j + n] += W[0, 0, m, n] * grad
            current_step.set(k + 1)

    @output
    @render.ui
    def dz0_display():
        k = current_step()
        highlight = [(k // 3, k % 3)] if k < 9 else []
        return ui.HTML(matrix_to_html(dZ0, highlight, prefix="dZ0："))

    @output
    @render.ui
    def dz_display():
        k = current_step()
        if k == 0:
            return ui.HTML(matrix_to_html(dZ, prefix="当前 dZ："))
        i = (k - 1) // 3
        j = (k - 1) % 3
        updated = [(i + m, j + n) for m in range(2) for n in range(2)]
        return ui.HTML(matrix_to_html(dZ, updated, prefix=f"dZ（更新来自 dZ0[{i},{j}]）"))

app = App(app_ui, server)