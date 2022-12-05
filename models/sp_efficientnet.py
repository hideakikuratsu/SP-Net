import math
from functools import partial

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from utils.config import FLAGS
from .slimmable_ops import SConv2d, SLinear, SSqueezeExcitation
from .slimmable_ops import SwitchableBatchNorm2d, SPBatchNorm2d
from .slimmable_ops import make_divisible


class MBConv(nn.Module):

    def __init__(
            self, inp, outp, kernel_size, stride, expand_ratio, n,
            stochastic_depth_prob, norm_layer
    ):

        global conv_m, bn_m, conv_ms, bn_ms, \
            block_conv_ms, block_bn_ms, major_idx, minor_idx, residual
        super().__init__()
        block_conv_ms = []
        block_bn_ms = []
        minor_idx = 0
        assert stride in [1, 2]
        self.major_idx = major_idx
        self.residual_connection = stride == 1 and inp == outp
        residual = self.residual_connection

        layers = []
        # expand
        expand_inp = [make_divisible(i * expand_ratio) for i in inp]
        if expand_ratio != 1:
            layers += [
                conv_m := SConv2d(
                    in_channels_list=inp, out_channels_list=expand_inp,
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                bn_m := norm_layer(expand_inp),
                nn.SiLU(inplace=True),
            ]
            set_module_link_in_block()

        # depthwise
        padding = (kernel_size - 1) // 2
        layers += [
            conv_m := SConv2d(
                in_channels_list=expand_inp, out_channels_list=expand_inp,
                kernel_size=kernel_size, stride=stride, padding=padding,
                groups_list=expand_inp, bias=False
            ),
            bn_m := norm_layer(expand_inp),
            nn.SiLU(inplace=True)
        ]
        set_module_link_in_block()

        # squeeze and excitation
        squeeze_channels_list = [max(1, i // 4) for i in inp]
        layers += [
            se_m := SSqueezeExcitation(
                in_channels_list=expand_inp, squeeze_channels_list=squeeze_channels_list,
                activation=partial(nn.SiLU, inplace=True)
            )
        ]
        se_m.prev = [conv_m]

        # project back
        layers += [
            conv_m := SConv2d(
                in_channels_list=expand_inp, out_channels_list=outp,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            bn_m := norm_layer(outp)
        ]
        if self.residual_connection:
            conv_m.end_of_block = True
        elif n > 1:
            conv_m.need_mask = True
        set_module_link_in_block()

        self.body = nn.Sequential(*layers)
        if self.residual_connection:
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        conv_ms.append(block_conv_ms)
        bn_ms.append(block_bn_ms)

    def forward(self, x):

        if not FLAGS.zpm_pruning:
            res = self.body(x)
            if self.residual_connection:
                res = self.stochastic_depth(res)
                res += x
        elif not FLAGS.reinit_params:
            if FLAGS.idx == len(FLAGS.width_mult_list) - 1:
                res = self.body(x)
                if self.residual_connection:
                    prev_last_conv_m = FLAGS.conv_ms[self.major_idx - 1][-1]
                    if getattr(prev_last_conv_m, 'need_mask', False):
                        shortcut_restore_sort_idx = \
                            prev_last_conv_m.restore_sort_idx_list[FLAGS.idx]
                    else:
                        shortcut_restore_sort_idx = None
                    shortcut = x
                    if shortcut_restore_sort_idx is not None:
                        shortcut = shortcut[:, shortcut_restore_sort_idx, ...]

                    base_restore_sort_idx = \
                        FLAGS.conv_ms[self.major_idx][-1].restore_sort_idx_list[FLAGS.idx]
                    res = res[:, base_restore_sort_idx, ...]
                    res = self.stochastic_depth(res)
                    res += shortcut
            else:
                base = self.body(x)
                if self.residual_connection:
                    prev_last_conv_m = FLAGS.conv_ms[self.major_idx - 1][-1]
                    if getattr(prev_last_conv_m, 'need_mask', False):
                        shortcut_restore_sort_idx = \
                            prev_last_conv_m.restore_sort_idx_list[FLAGS.idx]
                        shortcut_mask = prev_last_conv_m.masks[FLAGS.idx]
                    else:
                        shortcut_restore_sort_idx = None
                        shortcut_mask = prev_last_conv_m.fused_masks[FLAGS.idx]
                    shortcut = x
                    if shortcut_restore_sort_idx is not None:
                        shortcut = shortcut[:, shortcut_restore_sort_idx, ...]

                    last_conv_m = FLAGS.conv_ms[self.major_idx][-1]
                    base_restore_sort_idx = last_conv_m.restore_sort_idx_list[FLAGS.idx]
                    base_mask = last_conv_m.masks[FLAGS.idx]
                    res = torch.zeros(
                        base.size(0), len(last_conv_m.fused_masks[FLAGS.idx]),
                        base.size(2), base.size(3), dtype=base.dtype, device=base.device
                    )
                    base = base[:, base_restore_sort_idx, ...]
                    base = self.stochastic_depth(base)
                    res[:, base_mask.bool(), ...] = base
                    res[:, shortcut_mask.bool(), ...] += shortcut
                    res = res[:, last_conv_m.fused_masks[FLAGS.idx].bool(), ...]
                else:
                    res = base
        else:
            if FLAGS.idx == len(FLAGS.width_mult_list) - 1:
                res = self.body(x)
                if self.residual_connection:
                    res = self.stochastic_depth(res)
                    res += x
            else:
                base = self.body(x)
                if self.residual_connection:
                    shortcut = x
                    last_conv_m = FLAGS.conv_ms[self.major_idx][-1]
                    res = torch.zeros(
                        base.size(0), len(last_conv_m.fused_masks[FLAGS.idx]),
                        base.size(2), base.size(3), dtype=base.dtype, device=base.device
                    )
                    res = res[:, last_conv_m.fused_masks[FLAGS.idx].bool(), ...]
                    base = self.stochastic_depth(base)
                    res[:, :base.size(1), ...] = base
                    res[:, -shortcut.size(1):, ...] += shortcut
                else:
                    res = base
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()

        self.features = []
        # head
        assert input_size % 32 == 0

        norm_layer = SPBatchNorm2d if FLAGS.sp_model else SwitchableBatchNorm2d

        self.setting_dict_dict = {
            'b0': {
                'width_scale': 1.0, 'depth_scale': 1.0,
                'dropout': 0.2, 'norm_layer': norm_layer
            },
            'b1': {
                'width_scale': 1.0, 'depth_scale': 1.1,
                'dropout': 0.2, 'norm_layer': norm_layer
            },
            'b2': {
                'width_scale': 1.1, 'depth_scale': 1.2,
                'dropout': 0.3, 'norm_layer': norm_layer
            },
            'b3': {
                'width_scale': 1.2, 'depth_scale': 1.4,
                'dropout': 0.3, 'norm_layer': norm_layer
            },
            'b4': {
                'width_scale': 1.4, 'depth_scale': 1.8,
                'dropout': 0.4, 'norm_layer': norm_layer
            },
            'b5': {
                'width_scale': 1.6, 'depth_scale': 2.2, 'dropout': 0.4,
                'norm_layer': partial(norm_layer, eps=0.001, momentum=0.01)
            },
            'b6': {
                'width_scale': 1.8, 'depth_scale': 2.6, 'dropout': 0.5,
                'norm_layer': partial(norm_layer, eps=0.001, momentum=0.01)
            },
            'b7': {
                'width_scale': 2.0, 'depth_scale': 3.1, 'dropout': 0.5,
                'norm_layer': partial(norm_layer, eps=0.001, momentum=0.01)
            }
        }
        self.setting_dict = self.setting_dict_dict[FLAGS.type]
        norm_layer = self.setting_dict['norm_layer']
        self.block_setting = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],
            [6,  24, 2, 2, 3],
            [6,  40, 2, 2, 5],
            [6,  80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        for setting in self.block_setting:
            setting[2] = int(math.ceil(setting[2] * self.setting_dict['depth_scale']))

        c = make_divisible(32 * self.setting_dict['width_scale'])
        channels = [make_divisible(c * width_mult) for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=[3] * len(channels), out_channels_list=channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                ),
                bn_m := norm_layer(channels),
                nn.SiLU(inplace=True),
            )
        )
        conv_ms = []
        bn_ms = []
        conv_m.prev = [None]
        bn_m.prev = [None]
        conv_m.idx = (0, -1)
        bn_m.idx = (0, -1)
        conv_ms.append(conv_m)
        bn_ms.append(bn_m)

        # building inverted residual blocks
        total_stage_blocks = sum([setting[2] for setting in self.block_setting])
        stochastic_depth_prob = FLAGS.stochastic_depth_prob
        stage_block_id = 0
        cfg = [channels[-1]]
        major_idx = 0
        for t, c, n, s, k in self.block_setting:
            c = make_divisible(c * self.setting_dict['width_scale'])
            outp = [make_divisible(c * width_mult) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                major_idx += 1
                block = []
                expand_inp = make_divisible(channels[-1] * t)
                if t != 1:
                    block.append(expand_inp)
                block += [expand_inp, outp[-1]]

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                self.features.append(
                    MBConv(
                        inp=channels, outp=outp, kernel_size=k, stride=(s if i == 0 else 1),
                        expand_ratio=t, n=n, stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer
                    )
                )
                is_identity_connection = (s if i == 0 else 1) == 1 and channels == outp
                block.append(is_identity_connection)
                cfg.append(block)

                stage_block_id += 1
                channels = outp

        # tail
        major_idx += 1
        self.outp = [make_divisible(4 * channels[-1] * width_mult)
                     if width_mult > 1.0 else 4 * channels[-1]
                     for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=channels, out_channels_list=self.outp,
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                bn_m := norm_layer(self.outp),
                nn.SiLU(inplace=True),
            )
        )
        # conv_m.no_prune = True
        conv_m.prev = [conv_ms[-1][-1]]
        bn_m.prev = [bn_ms[-1][-1]]
        conv_ms[-1][-1].next = [conv_m]
        bn_ms[-1][-1].next = [bn_m]
        conv_m.idx = (major_idx, -1)
        bn_m.idx = (major_idx, -1)
        conv_ms.append(conv_m)
        bn_ms.append(bn_m)

        cfg.append(self.outp[-1])
        FLAGS.cfg = cfg
        FLAGS.cfg_only_bn = [c for c in cfg if c != 'M']

        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.setting_dict['dropout'], inplace=True),
            linear_m := SLinear(self.outp, [num_classes] * len(self.outp)),
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (SPBatchNorm2d, nn.GroupNorm)):
                if FLAGS.zero_gamma and getattr(m, 'end_of_block', False):
                    nn.init.zeros_(m.weight)
                else:
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
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)


def set_module_link_in_block():

    global conv_m, bn_m, conv_ms, bn_ms, block_conv_ms, block_bn_ms, major_idx, minor_idx
    if len(block_conv_ms) == 0:
        prev_conv_m = conv_ms[-1]
        prev_bn_m = bn_ms[-1]
    else:
        prev_conv_m = block_conv_ms[-1]
        prev_bn_m = block_bn_ms[-1]

    if isinstance(prev_conv_m, list):
        prev_conv_m = prev_conv_m[-1]
        prev_bn_m = prev_bn_m[-1]
    conv_m.prev = [prev_conv_m]
    bn_m.prev = [prev_bn_m]
    prev_conv_m.next = [conv_m]
    prev_bn_m.next = [bn_m]

    conv_m.idx = (major_idx, minor_idx)
    bn_m.idx = (major_idx, minor_idx)
    if not conv_m.groups_list == [1] * len(conv_m.groups_list):
        conv_m.out_channels_list = conv_m.prev[0].out_channels_list
        conv_m.groups_list = conv_m.prev[0].out_channels_list
        bn_m.num_features_list = bn_m.prev[0].num_features_list
    if not FLAGS.zpm_pruning:
        if getattr(conv_m, 'end_of_block', False):
            conv_m.out_channels_list = conv_ms[-1][-1].out_channels_list
            bn_m.num_features_list = bn_ms[-1][-1].num_features_list
    block_conv_ms.append(conv_m)
    block_bn_ms.append(bn_m)

    minor_idx += 1
