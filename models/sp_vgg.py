import torch.nn as nn

from utils.config import FLAGS
from .slimmable_ops import SConv2d, SLinear
from .slimmable_ops import SwitchableBatchNorm2d, SPBatchNorm2d


class Model(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, num_classes=1000, input_size=224):

        global conv_m, bn_m, conv_ms, bn_ms, major_idx
        super().__init__()

        norm_layer = SPBatchNorm2d if FLAGS.sp_model else SwitchableBatchNorm2d

        conv_ms = []
        bn_ms = []
        self.features = []
        assert input_size % 16 == 0
        channels = [3 for _ in FLAGS.width_mult_list]
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
               512, 512, 512, 512, 'M', 512, 512, 512, 512]
        major_idx = 0
        for v in cfg:
            if v == 'M':
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            else:
                outp = [int(v * width_mult) for width_mult in FLAGS.width_mult_list]
                self.features += [
                    conv_m := SConv2d(
                        in_channels_list=channels, out_channels_list=outp,
                        kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    bn_m := norm_layer(outp),
                    nn.ReLU(inplace=True)
                ]
                channels = outp
                set_module_link()
        FLAGS.cfg = cfg
        FLAGS.cfg_only_bn = [c for c in cfg if c != 'M']

        avg_pool_size = input_size // 16
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        # noinspection PyUnboundLocalVariable
        self.classifier = nn.Sequential(
            linear_m := SLinear(outp, [num_classes for _ in outp])
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
    conv_ms.append(conv_m)
    bn_ms.append(bn_m)

    major_idx += 1
