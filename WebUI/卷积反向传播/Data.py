import torch
import torch.nn.functional as F
from typing import Dict
from shiny import reactive

class Data:
    def __init__(self, d_H, d_W, d_C, f, seed):
        # 输入维度
        self.d_H = d_H
        self.d_W = d_W
        self.d_C = d_C
        self.f = f
        self.seed = seed

        # 输入、卷积核和输出和梯度
        torch.manual_seed(seed)
        self.Z = torch.randint(
            low=-5, high=6, size=(1, self.d_C, self.d_H, self.d_W), dtype=torch.float32, requires_grad=True)
        self.W = torch.randint(
            low=-3, high=4, size=(1, self.d_C, self.f, self.f), dtype=torch.float32, requires_grad=True)
        self.Z0 = F.conv2d(self.Z, self.W)
        self.dZ0 = torch.randint(low=-3, high=3, size=self.Z0.shape, dtype=torch.float32)
        self.Z0.backward(self.dZ0)
        self.dZ = self.Z.grad
        self.dW = self.W.grad

        # 输出维度
        self.d_H_l = self.Z0.shape[2]
        self.d_W_l = self.Z0.shape[3]
        self.d_C_l = 1
        self._elems_per_chan = self.d_H_l * self.d_W_l



class DZCalculator():
        def __init__(self, data_instance: Data):
            self.data = data_instance
            # dZ计算相关
            self._current_step_dZ = reactive.Value(0)
            self._dZ_cache = reactive.Value(
                torch.zeros_like(self.data.Z, dtype=torch.float32))
            self._dZ_cache_last = reactive.Value(
                torch.zeros_like(self.data.Z, dtype=torch.float32))
            self._last_slice_accumulated = reactive.Value(False)

        # dZ 计算相关方法
        @property
        def current_step_dZ(self):
            return self._current_step_dZ.get()

        @property
        def dZ_cache(self):
            return self._dZ_cache.get()

        @property
        def dZ_cache_last(self):
            return self._dZ_cache_last.get()
        
        @property
        def last_slice_accumulated(self):
            return self._last_slice_accumulated.get()

        @current_step_dZ.setter
        def current_step_dZ(self, value):
            if value <= self.data._elems_per_chan:
                self._current_step_dZ.set(value)
            else:
                pass

        def get_focus_ij(self) -> list:
            if self.current_step_dZ == 0:
                return []
            elif self.current_step_dZ > self.data.d_H_l * self.data.d_W_l:
                return []
            else:
                idx = self.current_step_dZ - 1
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
            if self.current_step_dZ == self.data._elems_per_chan:
                self._last_slice_accumulated.set(True)

        def reset_dZ_cache(self):
            self._dZ_cache.set(
                torch.zeros_like(self.Z, dtype=torch.float32))
            self._dZ_cache_last.set(
                torch.zeros_like(self.Z, dtype=torch.float32))
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