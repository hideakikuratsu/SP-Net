import torch.nn as nn

from utils.config import FLAGS
from .slimmable_ops import SwitchableBatchNorm2d, SPBatchNorm2d
from .slimmable_ops import SConv2d, SLinear


class DepthwiseSeparableConv(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, inp, outp, stride, norm_layer):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()
        assert stride in [1, 2]

        layers = [
            conv_m := SConv2d(
                in_channels_list=inp, out_channels_list=inp,
                kernel_size=3, stride=stride, padding=1, groups_list=inp, bias=False
            ),
            bn_m := norm_layer(inp),
            nn.ReLU6(inplace=True)
        ]
        set_module_link()

        layers += [
            conv_m := SConv2d(
                in_channels_list=inp, out_channels_list=outp,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            bn_m := norm_layer(outp),
            nn.ReLU6(inplace=True)
        ]
        set_module_link()
        self.body = nn.Sequential(*layers)

    def forward(self, x):

        return self.body(x)


class Model(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, num_classes=1000, input_size=224):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()

        norm_layer = SPBatchNorm2d if FLAGS.sp_model else SwitchableBatchNorm2d

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = [int(32 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=[3 for _ in range(len(channels))],
                    out_channels_list=channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                ),
                bn_m := norm_layer(channels),
                nn.ReLU6(inplace=True))
        )
        conv_ms = []
        bn_ms = []
        conv_m.prev = [None]
        bn_m.prev = [None]
        conv_m.idx = (0, -1)
        bn_m.idx = (0, -1)
        conv_ms.append(conv_m)
        bn_ms.append(bn_m)

        # body
        cfg = [channels[-1]]
        major_idx = 1
        for c, n, s in self.block_setting:
            outp = [int(c * width_mult) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                cfg.extend([channels[-1], outp[-1]])
                self.features.append(
                    DepthwiseSeparableConv(
                        inp=channels, outp=outp, stride=(s if i == 0 else 1), norm_layer=norm_layer
                    )
                )
                channels = outp
        FLAGS.cfg = cfg
        FLAGS.cfg_only_bn = [c for c in cfg if c != 'M']

        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        # noinspection PyUnboundLocalVariable
        self.classifier = nn.Sequential(
            linear_m := SLinear(outp, [num_classes for _ in range(len(outp))])
        )
        linear_m.prev = [bn_ms[-1]]
        conv_ms[-1].next = [None]
        bn_ms[-1].next = [None]
        FLAGS.conv_ms = conv_ms
        FLAGS.bn_ms = bn_ms

        if FLAGS.reset_parameters:
            init_bn_val = FLAGS.init_bn_val if FLAGS.sp_model else 1.0
            self.reset_parameters(init_bn_val=init_bn_val)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def reset_parameters(self, init_bn_val=0.5, reinit=False):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming normal initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, SPBatchNorm2d):
                nn.init.constant_(m.weight, init_bn_val)
                nn.init.zeros_(m.bias)
                if reinit:
                    m.reset_running_stats()
                    m.bn = nn.ModuleList(
                        [nn.BatchNorm2d(i, affine=False) for i in m.num_features_list[:-1]]
                    )
            elif isinstance(m, SwitchableBatchNorm2d):
                for bn in m.bn:
                    nn.init.constant_(bn.weight, init_bn_val)
                    nn.init.zeros_(bn.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def set_module_link():

    global conv_m, bn_m, conv_ms, bn_ms, major_idx
    if len(conv_ms) == 0:
        prev_conv_m = None
        prev_bn_m = None
    else:
        prev_conv_m = conv_ms[-1]
        prev_bn_m = bn_ms[-1]

    conv_m.prev = [prev_conv_m]
    bn_m.prev = [prev_bn_m]
    if len(conv_ms) != 0:
        prev_conv_m.next = [conv_m]
        prev_bn_m.next = [bn_m]

    conv_m.idx = (major_idx, -1)
    bn_m.idx = (major_idx, -1)
    if not conv_m.groups_list == [1] * len(conv_m.groups_list):
        conv_m.out_channels_list = conv_ms[-1].out_channels_list
        conv_m.groups_list = conv_ms[-1].out_channels_list
        bn_m.num_features_list = bn_ms[-1].num_features_list
    conv_ms.append(conv_m)
    bn_ms.append(bn_m)

    major_idx += 1
