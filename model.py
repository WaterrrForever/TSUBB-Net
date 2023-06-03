import torch
import torch.nn as nn
import torch.nn.functional as F

from shape import yuyi


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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


class Block_Transpose(nn.Module):
    def __init__(self, dim, drop_rate=0, layer_scale_init_value=1e-6):
        super().__init__()
        self.td_conv1 = nn.Linear(dim, 4 * dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((4 * dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.td_conv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()
        self.td_dwconv = nn.ConvTranspose2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.td_conv1(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.drop_path(x)
        x = self.td_conv2(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, C, H, W] -> [N, H, W, C]
        x = self.td_dwconv(x)
        x = self.norm(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_chans: int = 1, depths: list = None, dims: list = None, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6, ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                     sum(depths))]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])],
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_e = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x_e.append(x)
        # self.x_e = torch.tensor([item.cpu().detach().numpy() for item in self.x_e])
        return x_e


class Decoder(nn.Module):
    def __init__(self, out_chans: int = 1, depths: list = None, dims: list = None, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6, ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        stem = nn.Sequential(LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
                             nn.ConvTranspose2d(dims[3], out_chans, kernel_size=4, stride=4))
        self.downsample_layers.append(stem)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                     sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[Block_Transpose(dim=dims[i], drop_rate=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        # self.x_d = []

    # def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
    def forward(self, x):
        if type(x) == list:
            x_d = []
            for i in range(4):
                x_mid = x[3 - i].clone()
                for j in range(4 - i):
                    x_mid = self.stages[j + i](x_mid)
                    x_mid = self.downsample_layers[j + i](x_mid)
                x_d.append(x_mid)
            return x_d
        elif type(x) == torch.Tensor:
            for j in range(3):
                x = self.stages[j + 1](x)
                x = self.downsample_layers[j + 1](x)
            return x


class shape(nn.Module):
    def __init__(self, dims: list = None, dims1: list = None):
        super().__init__()
        # self.yuyi = yuyi()
        self.stages = nn.Sequential(
            *[yuyi(in_channels=dims[i], out_channels=dims1[i]) for i in range(4)]
        )

    def forward(self, x):
        x_s = []
        for i in range(4):
            x_mid = x[i].clone()
            x_mid = self.stages[i](x_mid)
            x_s.append(x_mid)
        return x_s


class Waterrr(nn.Module):
    def __init__(self, out_ch: int = 1):
        super().__init__()
        self.encoder = Encoder(depths=[2, 6, 2, 2], dims=[32, 64, 128, 256])
        self.decoder = Decoder(depths=[2, 2, 6, 2], dims=[256, 128, 64, 32])
        self.shape = shape(dims=[32, 64, 128, 256], dims1=[16, 32, 64, 128])
        self.apply(self._init_weights)
        self.out_conv = nn.Conv2d(4, out_ch, kernel_size=1)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):

        x_1 = self.encoder(x)
        x_2 = self.decoder(x_1)
        x_3 = self.out_conv(torch.concat(x_2, dim=1))
        x_2.append(x_3)
        x_shape = self.shape(x_1)
        x_shape = self.decoder(x_shape)
        x_shape1 = self.out_conv(torch.concat(x_shape, dim=1))
        x_shape.append(x_shape1)
        for i in range(len(x_2)):
            x_2[i] = F.gelu(x_2[i])
            x_shape[i] = F.gelu(x_shape[i])
        return x_2, x_1, x_shape


class Waterrr_pre(nn.Module):
    def __init__(self, out_ch: int = 1):
        super().__init__()
        self.encoder = Encoder(depths=[1, 3, 1, 1], dims=[32, 64, 128, 256])
        self.decoder = Decoder(depths=[1, 1, 3, 1], dims=[256, 128, 64, 32])
        self.shape = shape(dims=[32, 64, 128, 256], dims1=[16, 32, 64, 128])
        self.apply(self._init_weights)  # 初始化权重
        self.out_conv = nn.Conv2d(4, out_ch, kernel_size=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):

        x_1 = self.encoder(x)
        x_2 = self.decoder(x_1)
        x_3 = self.out_conv(torch.concat(x_2, dim=1))
        x_2.append(x_3)
        x_shape = self.shape(x_1)
        x_shape = self.decoder(x_shape)
        x_shape1 = self.out_conv(torch.concat(x_shape, dim=1))
        x_shape.append(x_shape1)
        for i in range(len(x_2)):
            x_2[i] = F.gelu(x_2[i])
            x_shape[i] = F.gelu(x_shape[i])
        return x_2, x_shape
