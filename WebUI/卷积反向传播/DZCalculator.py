import torch
from typing import Dict
from shiny import reactive
from Data import Data

class DZCalculator():
    def __init__(self, data_instance: Data):
        self.data = data_instance
        self._steps = reactive.Value(0)
        self._dZ_cache = reactive.Value(
            torch.zeros_like(self.data.Z, dtype=torch.float32))
        self._dZ_cache_last = reactive.Value(
            torch.zeros_like(self.data.Z, dtype=torch.float32))
        self._last_slice_accumulated = reactive.Value(False)

    @property
    def steps(self):
        return self._steps.get()

    @property
    def dZ_cache(self):
        return self._dZ_cache.get()

    @property
    def dZ_cache_last(self):
        return self._dZ_cache_last.get()
    
    @property
    def last_slice_accumulated(self):
        return self._last_slice_accumulated.get()

    @steps.setter
    def steps(self, value):
        if value <= self.data._elems_per_chan:
            self._steps.set(value)
        else:
            pass

    def get_focus_ij(self) -> list:
        if self.steps == 0:
            return []
        elif self.steps > self.data.d_H_l * self.data.d_W_l:
            return []
        else:
            idx = self.steps - 1
            in_channel_idx = idx % self.data._elems_per_chan
            c = idx // self.data._elems_per_chan
            i = in_channel_idx // self.data.d_W_l
            j = in_channel_idx % self.data.d_W_l
        return [(c, i, j)]

    def get_focus_i(self):
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, i, _ = coords[0]
        return i

    def get_focus_j(self):
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, _, j = coords[0]
        return j

    def get_Z_slice_ij(self):
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, i, j = coords[0]
        return self.data.Z[:, :, i:i+self.data.f, j:j+self.data.f]

    def get_Z0_ij(self):
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, i, j = coords[0]
        return self.data.Z0[:, :, i:i+1, j:j+1]

    def get_dZ0_ij(self):
        coords = self.get_focus_ij()
        if not coords:
            return None
        _, i, j = coords[0]
        return self.data.dZ0[:, :, i:i+1, j:j+1]

    def get_dZ_slice_ij(self):
        return self.get_dZ0_ij() * self.data.W
    
    def add_to_dZ_cache(self, dZ_slice_val):
        # 在累加前，将当前的 dZ_cache 状态保存到 _dZ_cache_last
        self._dZ_cache_last.set(self._dZ_cache.get().clone())
        coords = self.get_focus_ij()
        if not coords:
            return
        c, i, j = coords[0]

        current_dZ_cache = self._dZ_cache.get().clone()
        # 将 dZ_slice_val 加到 dZ_cache 的相应区域
        current_dZ_cache[:, :, i:i + self.data.f, j:j + self.data.f] += dZ_slice_val
        self._dZ_cache.set(current_dZ_cache)

        # 如果当前步数是最后一个有效步数，则设置标志位
        if self.steps == self.data._elems_per_chan:
            self._last_slice_accumulated.set(True)

    def reset_dZ_cache(self):
        self._dZ_cache.set(
            torch.zeros_like(self.data.Z, dtype=torch.float32))
        self._dZ_cache_last.set(
            torch.zeros_like(self.data.Z, dtype=torch.float32))
        self._last_slice_accumulated.set(False) # 重置标志位

    def get_highlight_Z0(self):
        coords = self.get_focus_ij()
        if not coords:
            return []
        c, i, j = coords[0]
        return [(self.data.d_C_l-1, i, j)]
    
    def get_highlight_Z(self):
        coords = self.get_focus_ij()
        if not coords:
            return []
        c, i, j = coords[0]
        highlights = []
        # 对所有 channel 逐个添加 (c, h, w) 坐标
        for c_idx in range(self.data.d_C):
            for di in range(self.data.f):
                for dj in range(self.data.f):
                    highlights.append((c_idx, i + di, j + dj))
        return highlights