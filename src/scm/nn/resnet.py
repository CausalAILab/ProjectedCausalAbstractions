import torch as T
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, h_size=128, group_layers=2, use_layer_norm=True):
        super(BasicBlock, self).__init__()
        self.h_size = h_size
        self.final_norm = None

        layers = []
        for l in range(group_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.h_size, self.h_size))
        if use_layer_norm:
            self.final_norm = nn.LayerNorm(self.h_size)
            layers.append(self.final_norm)

        self.final_relu = nn.ReLU()
        self.mlp_nn = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.mlp_nn(x)
        out += identity
        out = self.final_relu(out)

        return out


class MLP_ResNet_Module(nn.Module):
    def __init__(self, i_size, o_size, groups=1, width_per_group=64, group_layers=2,
                 use_layer_norm=True, zero_init_residual=False):
        super(MLP_ResNet_Module, self).__init__()

        self.groups = groups
        self.base_width = width_per_group
        layers = [nn.Linear(i_size, width_per_group)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(width_per_group))
        layers.append(nn.ReLU(inplace=True))

        for i in range(groups):
            layers.append(BasicBlock(h_size=width_per_group, group_layers=group_layers,
                                          use_layer_norm=use_layer_norm))
        layers.append(nn.Linear(width_per_group, o_size))

        self.mlp_resnet = nn.Sequential(*layers)
        self.device_param = nn.Parameter(T.empty(0))
        self.mlp_resnet.apply(self.init_weights)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.final_norm.weight, 0)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight,
                                     gain=T.nn.init.calculate_gain('relu'))

    def forward(self, x, include_inp=False):
        if include_inp:
            return self.mlp_resnet(x), x

        return self.mlp_resnet(x)
