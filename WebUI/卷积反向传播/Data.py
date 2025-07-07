import torch
import torch.nn.functional as F
from typing import Dict


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
    
    def get_focus_ij(self, steps: int) -> list:
        if steps == 0:
            return []
        elif steps > self.d_H_l * self.d_W_l:
            return []
        else:
            elems_per_channel = self.d_H * self.d_W
            idx = steps - 1
            in_channel_idx = idx % elems_per_channel
            c = idx // elems_per_channel
            i = in_channel_idx // self.d_W_l
            j = in_channel_idx % self.d_W_l

        return [(c, i, j)]
    
    def get_focus_i(self, steps):
        coords = self.get_focus_ij(steps)
        if not coords:
            return None
        _, i, _ = coords[0]
        return i

    def get_focus_j(self, steps):
        coords = self.get_focus_ij(steps)
        if not coords:
            return None
        _, _, j = coords[0]
        return j

    def get_Z_slice_ij(self, steps):
        coords = self.get_focus_ij(steps)
        if not coords:
            return None
        _, i, j = coords[0]
        return self.Z[..., i:i + self.f, j:j + self.f]
    
    def get_Z0_ij(self, steps):
        coords = self.get_focus_ij(steps)
        if not coords:
            return None
        _, i, j = coords[0]
        return self.Z0[..., i, j]
