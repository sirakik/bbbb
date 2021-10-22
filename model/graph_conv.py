import torch
import torch.nn as nn


# =============== Spatial Temporal Graph Convolution Block ===============
class STGCBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 s_kernel_size,
                 t_kernel_size,
                 dropout,
                 residual,
                 A_size,
                 bias=True):
        super().__init__()

        # Spatial Graph Convolution
        self.sgc = SpatialGC(in_channels=in_channels,
                             out_channels=out_channels,
                             s_kernel_size=s_kernel_size,
                             bias=bias)

        # Learnable weight matrix M
        self.M = nn.Parameter(torch.ones(A_size))

        # Temporal Graph Convolution unit
        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),
                                           (stride, 1),
                                           ((t_kernel_size - 1) // 2, 0),
                                           bias=bias),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout))

        # Residual
        if not residual:
            self.residual = lambda x: 0

        elif(in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1),
                                                    bias=bias),
                                          nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x, A):

        x = self.tgc(self.sgc(x, A * self.M)) + self.residual(x)
        x = self.relu(x)

        return x


# =============== Spatial Graph Convolution ===============
class SpatialGC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 bias):
        super().__init__()

        self.s_kernel_size = s_kernel_size

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * s_kernel_size,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=bias)

    def forward(self, x, A):

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()
