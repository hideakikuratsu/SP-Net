import torch.nn as nn
import torch.nn.functional as F

from utils.config import FLAGS


class SConv2d(nn.Conv2d):

    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=None, bias=True):

        if groups_list is None:
            groups_list = [1] * len(FLAGS.width_mult_list)
        super().__init__(
            in_channels_list[-1], out_channels_list[-1],
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups_list[-1], bias=bias
        )
        self.in_channels_list = in_channels_list.copy()
        self.out_channels_list = out_channels_list.copy()
        self.groups_list = groups_list.copy()
        self.width_mult = FLAGS.width_mult_list[-1]

    def forward(self, inputs):

        idx = FLAGS.idx
        self.in_channels = inputs.size(1)
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        if idx == len(FLAGS.width_mult_list) - 1:
            return F.conv2d(
                inputs, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups
            )

        if self.prev[0] is not None and self.groups == 1:
            end_of_block = getattr(self.prev[0], 'end_of_block', False)
            shortcut = getattr(self.prev[0], 'shortcut', False)
            if FLAGS.zpm_pruning and not FLAGS.reinit_params and (end_of_block or shortcut):
                end_of_block_idx = -1 if not shortcut else -2
                in_idx = FLAGS.conv_ms[self.idx[0] - 1][end_of_block_idx].fused_masks[idx].bool()
                weight = self.weight[:, in_idx, ...]
            elif not FLAGS.mask_pruning:
                weight = self.weight[:, :self.in_channels, ...]
            else:
                in_idx = self.prev[0].masks[idx].bool()
                weight = self.weight[:, in_idx, ...]
        else:
            weight = self.weight

        if not FLAGS.mask_pruning:
            weight = weight[:self.out_channels, ...]
        else:
            weight = weight[self.masks[idx].bool(), ...]

        if self.bias is not None:
            if not FLAGS.mask_pruning:
                bias = self.bias[:self.out_channels]
            else:
                bias = self.bias[self.masks[idx].bool()]
        else:
            bias = None

        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class SwitchableBatchNorm2d(nn.Module):

    def __init__(self, num_features_list, **kwargs):

        super().__init__()
        self.num_features_list = num_features_list
        self.num_features = num_features_list[-1]
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i, **kwargs))
        self.bn = nn.ModuleList(bns)
        self.width_mult = FLAGS.width_mult_list[-1]

    def forward(self, inputs):

        idx = FLAGS.idx
        self.num_features = self.num_features_list[idx]
        y = self.bn[idx](inputs)

        return y


class SPBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features_list, **kwargs):

        super().__init__(num_features_list[-1], **kwargs)
        self.num_features_list = num_features_list.copy()
        self.width_mult = FLAGS.width_mult_list[-1]
        self.bn = nn.ModuleList([nn.BatchNorm2d(i, affine=False) for i in num_features_list[:-1]])

    def forward(self, inputs):

        idx = FLAGS.idx
        self.num_features = self.num_features_list[idx]
        if idx == len(FLAGS.width_mult_list) - 1:
            return F.batch_norm(
                inputs, self.running_mean, self.running_var, self.weight, self.bias, self.training,
                self.momentum, self.eps
            )

        if not FLAGS.mask_pruning:
            weight = self.weight[:self.num_features]
            bias = self.bias[:self.num_features]
        else:
            conv_m = FLAGS.conv_ms[self.idx[0]]
            if isinstance(conv_m, list):
                conv_m = conv_m[self.idx[1]]
            mask = conv_m.masks[idx].bool()
            weight = self.weight[mask]
            bias = self.bias[mask]
        running_mean = self.bn[idx].running_mean
        running_var = self.bn[idx].running_var

        return F.batch_norm(
            inputs, running_mean, running_var, weight, bias, self.training,
            self.momentum, self.eps
        )


class SLinear(nn.Linear):

    def __init__(self, in_features_list, out_features_list, bias=True):

        super().__init__(in_features_list[-1], out_features_list[-1], bias=bias)
        self.in_features_list = in_features_list.copy()
        self.out_features_list = out_features_list.copy()
        self.width_mult = FLAGS.width_mult_list[-1]

    def forward(self, inputs):

        idx = FLAGS.idx
        self.in_features = inputs.size(1)
        if idx == len(FLAGS.width_mult_list) - 1:
            return F.linear(inputs, self.weight, self.bias)

        if FLAGS.zpm_pruning and not FLAGS.reinit_params and 'resnet' in FLAGS.model:
            in_idx = FLAGS.conv_ms[-1][-1].fused_masks[idx].bool()
            weight = self.weight[:, in_idx]
        elif not FLAGS.mask_pruning:
            weight = self.weight[:, :self.in_features]
        else:
            in_idx = FLAGS.conv_ms[-1][-1].masks[idx].bool()
            weight = self.weight[:, in_idx]

        if self.bias is not None:
            bias = self.bias
        else:
            bias = None

        return F.linear(inputs, weight, bias)


class SAvgPool2d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return F.avg_pool2d(inputs, inputs.size(-1))


class SSEConv2d(nn.Conv2d):

    def __init__(self, in_channels_list, out_channels_list):

        super().__init__(
            in_channels_list[-1], out_channels_list[-1],
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.in_channels_list = in_channels_list.copy()
        self.out_channels_list = out_channels_list.copy()
        self.width_mult = FLAGS.width_mult_list[-1]

    def forward(self, inputs):

        idx = FLAGS.idx
        self.in_channels = inputs.size(1)
        self.out_channels = self.out_channels_list[idx]

        weight = self.weight[:self.out_channels, :self.in_channels, ...]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = None

        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class SSqueezeExcitation(nn.Module):

    def __init__(
            self, in_channels_list, squeeze_channels_list,
            activation=nn.ReLU, scale_activation=nn.Sigmoid
    ):

        super().__init__()
        self.in_channels_list = in_channels_list.copy()
        self.squeeze_channels_list = squeeze_channels_list.copy()
        self.avgpool = SAvgPool2d()
        self.fc1 = SSEConv2d(
            in_channels_list=in_channels_list, out_channels_list=squeeze_channels_list
        )
        self.fc2 = SSEConv2d(
            in_channels_list=squeeze_channels_list, out_channels_list=in_channels_list
        )
        self.activation = activation()
        self.scale_activation = scale_activation()
        self.width_mult = FLAGS.width_mult_list[-1]

    def _scale(self, inputs):

        idx = FLAGS.idx
        self.in_channels = inputs.size(1)
        self.squeeze_channels = self.squeeze_channels_list[idx]

        scale = self.avgpool(inputs)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)

        return self.scale_activation(scale)

    def forward(self, inputs):

        scale = self._scale(inputs)
        return scale * inputs


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if isinstance(m, nn.BatchNorm2d):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if FLAGS.cumulative_bn_stats:
            m.momentum = None
