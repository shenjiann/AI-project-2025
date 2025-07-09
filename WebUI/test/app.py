from shiny import reactive, App, ui, render
import torch
import torch.nn.functional as F
# 假设你的 DataLoader 和 dZCalculator 类定义在 data_loader.py 和 dz_calculator.py 中
# from data_loader import DataLoader
# from dz_calculator import dZCalculator

# 假设你已经定义了 DataLoader 和 dZCalculator 类，如下面的简化版本：
class DataLoader:
    def __init__(self, d_H, d_W, d_C, f, seed):
        self.d_H = d_H
        self.d_W = d_W
        self.d_C = d_C
        self.f = f
        self.seed = seed
        # 简化初始化，实际应包含所有张量和维度的初始化
        torch.manual_seed(seed)
        self.Z = torch.randint(low=-5, high=6, size=(1, d_C, d_H, d_W), dtype=torch.float32)
        self.W = torch.randint(low=-3, high=4, size=(1, d_C, f, f), dtype=torch.float32)
        self.Z0 = F.conv2d(self.Z, self.W)
        self.dZ0 = torch.randint(low=-3, high=3, size=self.Z0.shape, dtype=torch.float32)
        # 假设 dZ 和 dW 也在 DataLoader 中初始化了
        self.dZ_true = self.Z # 简化表示
        self.dW_true = self.W # 简化表示

    def get_Z(self):
        return self.Z
    def get_W(self):
        return self.W
    def get_dZ0(self):
        return self.dZ0
    def get_d_H_l(self):
        return self.Z0.shape[2]
    def get_d_W_l(self):
        return self.Z0.shape[3]
    def get_elems_per_chan(self):
        return self.Z0.shape[2] * self.Z0.shape[3] # 示例

    # 添加 dZ0_at_ij 等方法
    def get_dZ0_at_ij(self, i, j):
        return self.dZ0[:, :, i:i+1, j:j+1] # 示例

class DZCalculator:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        # 在这里缓存常用数据，例如 self._W = self.data_loader.get_W()
        self._W = self.data_loader.get_W()
        self._Z = self.data_loader.get_Z()
        self._elems_per_chan = self.data_loader.get_elems_per_chan()
        self._d_W_l = self.data_loader.get_d_W_l()

        # 初始化 reactive 状态
        self._current_step_dZ: reactive.Value[int] = reactive.Value(0)
        self._dZ_cache: reactive.Value[torch.Tensor] = reactive.Value(
            torch.zeros_like(self._Z, dtype=torch.float32)
        )
        self._dZ_cache_last: reactive.Value[torch.Tensor] = reactive.Value(
            torch.zeros_like(self._Z, dtype=torch.float32)
        )
        self._last_slice_accumulated: reactive.Value[bool] = reactive.Value(False)


    @property
    def current_step_dZ(self) -> int:
        return self._current_step_dZ.get()

    @current_step_dZ.setter
    def current_step_dZ(self, value: int):
        if 0 <= value <= self._elems_per_chan:
            self._current_step_dZ.set(value)

    def get_focus_ij(self) -> list: # 简化返回类型
        if not (1 <= self.current_step_dZ <= self._elems_per_chan):
            return []
        idx = self.current_step_dZ - 1
        in_channel_idx = idx % self._elems_per_chan
        c = idx // self._elems_per_chan
        i = in_channel_idx // self._d_W_l
        j = in_channel_idx % self._d_W_l
        return [(c, i, j)]

    def get_dZ_slice_ij(self) -> torch.Tensor:
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, i, j = coords[0]
        dZ0_val = self.data_loader.get_dZ0_at_ij(i, j)
        return dZ0_val * self._W # 使用缓存的 W


# Shiny 应用骨架
app_ui = ui.page_fluid(
    ui.input_numeric("height", "Height", value=5),
    ui.input_numeric("width", "Width", value=5),
    ui.input_numeric("channel", "Channels", value=1),
    ui.input_numeric("size", "Kernel Size", value=3),
    ui.input_numeric("seed", "Seed", value=42),
    ui.output_text_verbatim("debug_output")
)

def server(input, output, session):
    # --- 数据生成 ---
    @reactive.calc
    def data():
        """创建 DataLoader 实例，它是反应式的"""
        return DataLoader(
            d_H=input.height(),
            d_W=input.width(),
            d_C=input.channel(),
            f=input.size(),
            seed=input.seed()
        )

    # --- DZCalculator 生成 ---
    @reactive.calc
    def dz_calculator():
        """创建 DZCalculator 实例，它依赖于 data()"""
        return DZCalculator(data_loader=data()) # 将 data() reactive 对象传递给 DZCalculator 的构造函数

    # --- DWCalculator 生成 (类似地) ---
    @reactive.calc
    def dw_calculator():
        """创建 DWCalculator 实例，它也依赖于 data()"""
        # 假设你有一个 DWCalculator 类，它的初始化也接收 data_loader
        # return DWCalculator(data_loader=data())
        return f"DWCalculator would be initialized here with data() object." # 占位符

    @output
    @render.text
    def debug_output():
        # 可以在这里打印一些调试信息，确保对象被正确创建和更新
        # 访问 reactive 值时需要调用它们，例如 data() 或 dz_calculator()
        current_data_id = id(data())
        current_dz_calc_id = id(dz_calculator())

        # 示例：尝试从 dz_calculator 访问数据
        z_from_dz_calc = dz_calculator().data_loader.get_Z()

        # 示例：设置 dz_calculator 的步数并获取切片
        # 注意：reactive.Value 只能在 reactive.Effect 或 reactive.calc 中修改
        # 或者在输入事件处理函数中修改
        # 如果你想在非 reactive 上下文中修改 dz_calculator 内部的 reactive.Value，需要特殊处理
        # 这里仅为演示获取值
        if dz_calculator().current_step_dZ == 0: # 避免在每次渲染时都尝试设置
            # 通常你会在一个 input.observe 或其他 reactive 表达式中改变这个值
            pass # Skipping setting for now, as it's not the main focus of the question

        focus_ij = dz_calculator().get_focus_ij()
        dZ_slice = dz_calculator().get_dZ_slice_ij()


        return (
            f"DataLoader object ID: {current_data_id}\n"
            f"DZCalculator object ID: {current_dz_calc_id}\n"
            f"Z from DZCalculator's DataLoader: \n{z_from_dz_calc}\n"
            f"Focus i,j for DZCalculator: {focus_ij}\n"
            f"Calculated dZ slice (example): \n{dZ_slice}\n"
        )


app = App(app_ui, server)