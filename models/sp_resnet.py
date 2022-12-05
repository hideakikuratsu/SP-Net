import torch
import torch.nn as nn

from .slimmable_ops import SwitchableBatchNorm2d, SPBatchNorm2d
from .slimmable_ops import SConv2d, SLinear
from utils.config import FLAGS


class Block(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, inp, outp, stride, norm_layer):

        global conv_m, bn_m, conv_ms, bn_ms, block_conv_ms, block_bn_ms, major_idx, minor_idx, residual
        super().__init__()
        block_conv_ms = []
        block_bn_ms = []
        minor_idx = 0
        assert stride in [1, 2]
        self.major_idx = major_idx
        self.residual_connection = stride == 1 and inp == outp
        residual = self.residual_connection

        midp = [i // 4 for i in outp]
        layers = [
            conv_m := SConv2d(
                in_channels_list=inp, out_channels_list=midp,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            bn_m := norm_layer(midp),
            nn.ReLU(inplace=True)
        ]
        set_module_link_in_block()

        layers += [
            conv_m := SConv2d(
                in_channels_list=midp, out_channels_list=midp,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            bn_m := norm_layer(midp),
            nn.ReLU(inplace=True)
        ]
        set_module_link_in_block()

        layers += [
            conv_m := SConv2d(
                in_channels_list=midp, out_channels_list=outp,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            bn_m := norm_layer(outp)
        ]
        conv_m.end_of_block = True
        set_module_link_in_block()
        self.body = nn.Sequential(*layers)

        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=inp, out_channels_list=outp,
                    kernel_size=1, stride=stride, padding=0, bias=False
                ),
                bn_m := norm_layer(outp),
            )
            conv_m.shortcut = True
            set_module_link_in_block(is_shortcut=True)
        self.post_relu = nn.ReLU(inplace=True)

        conv_ms.append(block_conv_ms)
        bn_ms.append(block_bn_ms)

    def forward(self, x):

        if not FLAGS.zpm_pruning:
            res = self.body(x)
            if self.residual_connection:
                res += x
            else:
                res += self.shortcut(x)
        elif not FLAGS.reinit_params:
            if FLAGS.idx == len(FLAGS.width_mult_list) - 1:
                base_restore_sort_idx = \
                    FLAGS.conv_ms[self.major_idx][2].restore_sort_idx_list[FLAGS.idx]
                base = self.body(x)
                if self.residual_connection:
                    shortcut_restore_sort_idx = None
                    shortcut = x
                else:
                    shortcut_restore_sort_idx = \
                        FLAGS.conv_ms[self.major_idx][3].restore_sort_idx_list[FLAGS.idx]
                    shortcut = self.shortcut(x)
                base = base[:, base_restore_sort_idx, ...]
                if shortcut_restore_sort_idx is not None:
                    shortcut = shortcut[:, shortcut_restore_sort_idx, ...]
                res = base + shortcut
            else:
                base_restore_sort_idx = \
                    FLAGS.conv_ms[self.major_idx][2].restore_sort_idx_list[FLAGS.idx]
                base_mask = FLAGS.conv_ms[self.major_idx][2].masks[FLAGS.idx]
                base = self.body(x)
                if self.residual_connection:
                    shortcut_restore_sort_idx = None
                    shortcut_mask = FLAGS.conv_ms[self.major_idx - 1][2].fused_masks[FLAGS.idx]
                    shortcut = x
                else:
                    shortcut_restore_sort_idx = \
                        FLAGS.conv_ms[self.major_idx][3].restore_sort_idx_list[FLAGS.idx]
                    shortcut_mask = FLAGS.conv_ms[self.major_idx][3].masks[FLAGS.idx]
                    shortcut = self.shortcut(x)
                res = torch.zeros(
                    base.size(0), len(FLAGS.conv_ms[self.major_idx][2].fused_masks[FLAGS.idx]),
                    base.size(2), base.size(3),
                    dtype=base.dtype, device=base.device
                )
                base = base[:, base_restore_sort_idx, ...]
                if shortcut_restore_sort_idx is not None:
                    shortcut = shortcut[:, shortcut_restore_sort_idx, ...]
                res[:, base_mask.bool(), ...] = base
                res[:, shortcut_mask.bool(), ...] += shortcut
                res = res[:, FLAGS.conv_ms[self.major_idx][2].fused_masks[FLAGS.idx].bool(), ...]
        else:
            if FLAGS.idx == len(FLAGS.width_mult_list) - 1:
                res = self.body(x)
                if self.residual_connection:
                    res += x
                else:
                    res += self.shortcut(x)
            else:
                base = self.body(x)
                if self.residual_connection:
                    shortcut = x
                else:
                    shortcut = self.shortcut(x)
                res = torch.zeros(
                    base.size(0),
                    len(FLAGS.conv_ms[self.major_idx][2].fused_masks[FLAGS.idx]),
                    base.size(2), base.size(3),
                    dtype=base.dtype, device=base.device
                )
                res = res[:, FLAGS.conv_ms[self.major_idx][2].fused_masks[FLAGS.idx].bool(), ...]
                res[:, :base.size(1), ...] = base
                res[:, -shortcut.size(1):, ...] += shortcut
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, num_classes=1000, input_size=224):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()

        norm_layer = SPBatchNorm2d if FLAGS.sp_model else SwitchableBatchNorm2d

        self.features = []
        # head
        assert input_size % 32 == 0

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]

        feats = [64, 128, 256, 512]
        channels = [int(64 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.features.append(
            nn.Sequential(
                conv_m := SConv2d(
                    in_channels_list=[3] * len(channels), out_channels_list=channels,
                    kernel_size=7, stride=2, padding=3, bias=False
                ),
                bn_m := norm_layer(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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
        cfg = [channels[-1], 'M']
        major_idx = 0
        for stage_id, n in enumerate(self.block_setting):
            outp = [int(feats[stage_id] * width_mult * 4) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                major_idx += 1
                mid = outp[-1] // 4
                block = [mid, mid, outp[-1]]
                self.features.append(
                    Block(
                        inp=channels, outp=outp, stride=(2 if i == 0 and stage_id != 0 else 1),
                        norm_layer=norm_layer
                    )
                )
                is_identity_connection = False if i == 0 and stage_id != 0 \
                    else channels == outp
                if not is_identity_connection:
                    block.append(outp[-1])
                block.append(is_identity_connection)
                cfg.append(block)
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
            linear_m := SLinear(outp, [num_classes] * len(outp))
        )
        linear_m.prev = [bn_ms[-1][-1]]
        conv_ms[-1][-1].next = [None]
        bn_ms[-1][-1].next = [None]
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


def set_module_link_in_block(is_shortcut=False):

    global conv_m, bn_m, conv_ms, bn_ms, block_conv_ms, block_bn_ms, major_idx, minor_idx
    if len(block_conv_ms) == 0 or is_shortcut:
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
    if not is_shortcut:
        prev_conv_m.next = [conv_m]
        prev_bn_m.next = [bn_m]
    else:
        block_conv_ms[-1].next = [conv_m]
        block_bn_ms[-1].next = [bn_m]

    conv_m.idx = (major_idx, minor_idx)
    bn_m.idx = (major_idx, minor_idx)
    if not FLAGS.zpm_pruning:
        if getattr(conv_m, 'end_of_block', False) and residual:
            conv_m.out_channels_list = conv_ms[-1][-1].out_channels_list
            bn_m.num_features_list = bn_ms[-1][-1].num_features_list
        elif getattr(conv_m, 'shortcut', False):
            block_conv_ms[-1].out_channels_list = conv_m.out_channels_list
            block_bn_ms[-1].num_features_list = bn_m.num_features_list
    block_conv_ms.append(conv_m)
    block_bn_ms.append(bn_m)

    minor_idx += 1
