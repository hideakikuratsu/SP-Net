import torch
import torch.nn as nn

from utils.config import FLAGS
from .slimmable_ops import SwitchableBatchNorm2d, SPBatchNorm2d, SConv2d, SLinear
from .slimmable_ops import make_divisible


class InvertedResidual(nn.Module):

    def __init__(self, inp, outp, stride, expand_ratio, n, norm_layer):

        global conv_m, bn_m, conv_ms, bn_ms, block_conv_ms, block_bn_ms, major_idx, minor_idx, residual
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
        expand_inp = [i * expand_ratio for i in inp]
        if expand_ratio != 1:
            layers += [
                conv_m := SConv2d(
                    in_channels_list=inp, out_channels_list=expand_inp,
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                bn_m := norm_layer(expand_inp),
                nn.ReLU6(inplace=True),
            ]
            set_module_link_in_block()

        # depthwise
        layers += [
            conv_m := SConv2d(
                in_channels_list=expand_inp, out_channels_list=expand_inp,
                kernel_size=3, stride=stride, padding=1, groups_list=expand_inp, bias=False
            ),
            bn_m := norm_layer(expand_inp),
            nn.ReLU6(inplace=True)
        ]
        set_module_link_in_block()

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

        conv_ms.append(block_conv_ms)
        bn_ms.append(block_bn_ms)

    def forward(self, x):

        if not FLAGS.zpm_pruning:
            res = self.body(x)
            if self.residual_connection:
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
                    res[:, base_mask.bool(), ...] = base
                    res[:, shortcut_mask.bool(), ...] += shortcut
                    res = res[:, last_conv_m.fused_masks[FLAGS.idx].bool(), ...]
                else:
                    res = base
        else:
            if FLAGS.idx == len(FLAGS.width_mult_list) - 1:
                res = self.body(x)
                if self.residual_connection:
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
                    res[:, :base.size(1), ...] = base
                    res[:, -shortcut.size(1):, ...] += shortcut
                else:
                    res = base
                # base = self.body(x)
                # if self.residual_connection:
                #     prev_last_conv_m = FLAGS.conv_ms[self.major_idx - 1][-1]
                #     if getattr(prev_last_conv_m, 'need_mask', False):
                #         shortcut_mask = prev_last_conv_m.masks[FLAGS.idx]
                #     else:
                #         shortcut_mask = prev_last_conv_m.fused_masks[FLAGS.idx]
                #     shortcut = x
                #
                #     last_conv_m = FLAGS.conv_ms[self.major_idx][-1]
                #     base_mask = last_conv_m.masks[FLAGS.idx]
                #     res = torch.zeros(
                #         base.size(0), len(last_conv_m.fused_masks[FLAGS.idx]),
                #         base.size(2), base.size(3), dtype=base.dtype, device=base.device
                #     )
                #     res[:, base_mask.bool(), ...] = base
                #     res[:, shortcut_mask.bool(), ...] += shortcut
                #     res = res[:, last_conv_m.fused_masks[FLAGS.idx].bool(), ...]
                # else:
                #     res = base
        return res


class Model(nn.Module):

    def __init__(self, num_classes=1000, input_size=224):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()

        norm_layer = SPBatchNorm2d if FLAGS.sp_model else SwitchableBatchNorm2d

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = [make_divisible(32 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=[3] * len(channels), out_channels_list=channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                ),
                bn_m := norm_layer(channels),
                nn.ReLU6(inplace=True)
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

        # body
        cfg = [channels[-1]]
        major_idx = 0
        for t, c, n, s in self.block_setting:
            outp = [make_divisible(c * width_mult) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                major_idx += 1
                block = []
                expand_inp = channels[-1] * t
                if t != 1:
                    block.append(expand_inp)
                block += [expand_inp, outp[-1]]
                self.features.append(
                    InvertedResidual(
                        inp=channels, outp=outp, stride=(s if i == 0 else 1),
                        expand_ratio=t, n=n, norm_layer=norm_layer
                    )
                )
                is_identity_connection = (s if i == 0 else 1) == 1 and channels == outp
                block.append(is_identity_connection)
                cfg.append(block)
                channels = outp

        # tail
        # Cited implementation note from original paper
        # 'for multipliers less than one, we apply width multiplier to
        # all layers except the very last convolutional layer.
        # This improves performance for smaller models.'
        major_idx += 1
        self.outp = [make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
                     for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=channels, out_channels_list=self.outp,
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                bn_m := norm_layer(self.outp),
                nn.ReLU6(inplace=True),
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

        # classifier
        # Cited implementation note from original paper
        # 'for multipliers less than one, we apply width multiplier to
        # all layers except the very last convolutional layer.
        # This improves performance for smaller models.'
        self.classifier = nn.Sequential(
            linear_m := SLinear(self.outp, [num_classes] * len(self.outp))
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
                nn.init.normal_(m.weight, 0, 0.01)
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
