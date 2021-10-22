import torch
import torch.nn as nn

from model.graph import Graph
from model.graph_conv import STGCBlock



class ST_GCN(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 residual,
                 dropout,
                 t_kernel_size,
                 layout,
                 strategy,
                 hop_size):
        super().__init__()

        # Graph
        graph = Graph(layout=layout, strategy=strategy, hop_size=hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # STGC Block
        self.bn = nn.BatchNorm1d(in_channels, A.size(1))
        kwargs = dict(s_kernel_size=A.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A.size())
        self.stgc0 = STGCBlock(in_channels, 64, stride=1, dropout=0, **kwargs)
        self.stgc1 = STGCBlock(64, 64, stride=1, **kwargs)
        self.stgc2 = STGCBlock(64, 64, stride=1, **kwargs)
        self.stgc3 = STGCBlock(64, 64, stride=1, **kwargs)
        self.stgc4 = STGCBlock(64, 128, stride=2, **kwargs)
        self.stgc5 = STGCBlock(128, 128, stride=1, **kwargs)
        self.stgc6 = STGCBlock(128, 128, stride=1, **kwargs)
        self.stgc7 = STGCBlock(128, 256, stride=2, **kwargs)
        self.stgc8 = STGCBlock(256, 256, stride=1, **kwargs)
        self.stgc9 = STGCBlock(256, 256, stride=1, **kwargs)

        # prediction
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)

    def forward(self, x):

        # Batch Normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # STGC Block
        x = self.stgc0(x)
        x = self.stgc1(x)
        x = self.stgc2(x)
        x = self.stgc3(x)
        x = self.stgc4(x)
        x = self.stgc5(x)
        x = self.stgc6(x)
        x = self.stgc7(x)
        x = self.stgc8(x)
        x = self.stgc9(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
