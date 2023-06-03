import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            LayerNorm(out_channels),
            nn.GELU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels),
            nn.GELU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels),
            nn.GELU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ChannelWeighting(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_popl = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_popl(x).view(b, c)
        out = torch.abs(avg_out - max_out) * avg_out
        out = self.fc(out).view(b, c, 1, 1)
        x = x * out + x
        return x


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x.is_cuda == True:
            a = torch.zeros(x.shape).cuda()
        else:
            a = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            A = x[i]
            B = y[i]
            P = torch.reshape(A, (A.shape[0], -1))
            Q = torch.reshape(B, (B.shape[0], -1))
            S = torch.mm(Q.T, P)
            scaled_S = S / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float))
            attention_weights = F.softmax(scaled_S, dim=1)
            L = torch.matmul(Q, attention_weights)
            L = torch.reshape(torch.matmul(P, attention_weights), (B.shape[0], B.shape[1], -1))
            O = torch.add(L, B)
            a[i, :, :, :] = O
        return a


class yuyi(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ASPP = ASPP(in_channels, [3, 6, 9], in_channels)
        self.ChannelWeighting = ChannelWeighting(in_channels, out_channels)
        self.AttentionModule = AttentionModule()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        x1 = x
        x1 = self.conv1(x1)
        x = self.ChannelWeighting(x)
        x = self.ASPP(x)
        x = self.conv1(x)
        x2 = self.AttentionModule(x1, x)
        x2 = self.conv2(x2)
        return x2

# yuyi = yuyi(in_channels=256)
# a = torch.randn(16, 256, 7, 7)
# b = yuyi(a)
# print(b.shape)
