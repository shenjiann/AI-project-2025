import torch
import torch.nn.functional as F

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

    

