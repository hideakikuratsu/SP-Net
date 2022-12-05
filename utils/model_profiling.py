import numpy as np
import torch
import torch.nn as nn
import torchvision.ops

from models.slimmable_ops import SPBatchNorm2d, SAvgPool2d
from models.slimmable_ops import SwitchableBatchNorm2d
from utils.config import FLAGS

ignore_zeros_t = [
    nn.BatchNorm2d, nn.Dropout2d, nn.Dropout,
    nn.modules.padding.ZeroPad2d,
    nn.modules.activation.Sigmoid,
    torchvision.ops.StochasticDepth
]

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 10
params_space = 15
macs_space = 15
seconds_space = 11

num_forwards = FLAGS.num_forwards
pre_num_forwards = FLAGS.pre_num_forwards


class Timer(object):

    def __init__(self, verbose=False):

        self.verbose = verbose
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):

        self.start.record()
        return self

    def __exit__(self, *args):

        self.end.record()
        torch.cuda.synchronize()
        self.time = self.start.elapsed_time(self.end)
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(module):
    """get number of params in module"""
    bias = 0
    if isinstance(module, SwitchableBatchNorm2d):
        if module.width_mult == FLAGS.max_width:
            weight = sum([module.num_features_list[i] * 2 for i
                          in range(len(FLAGS.width_mult_list))])
        else:
            weight = module.num_features * 2
    elif isinstance(module, SPBatchNorm2d):
        weight = module.num_features * 2
    elif isinstance(module, nn.Conv2d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        weight = np.prod(
            [in_channels, out_channels, module.kernel_size[0], module.kernel_size[1]]
        )
        weight //= module.groups
        if module.bias is not None:
            bias = out_channels
    elif isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        weight = np.prod([in_features, out_features])

        if module.bias is not None:
            bias = out_features
    else:
        raise NotImplementedError(f'Not Implemented for {module.__repr__()}')

    return weight + bias


def run_forward(module, inputs):

    for _ in range(pre_num_forwards):
        module.forward(*inputs)
    with Timer() as t:
        for _ in range(num_forwards):
            module.forward(*inputs)

    return t.time * 1e6 / num_forwards


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])

    return name


def module_profiling(module, inputs, outputs, flops_and_params_only=False):

    ins = inputs[0].size()
    outs = outputs.size()

    idx = FLAGS.idx
    n_macs = getattr(module, 'n_macs', [0] * len(FLAGS.width_mult_list))
    n_params = getattr(module, 'n_params', [0] * len(FLAGS.width_mult_list))
    n_nano_seconds = getattr(module, 'n_nano_seconds', [0] * len(FLAGS.width_mult_list))
    setattr(module, 'n_macs', n_macs)
    setattr(module, 'n_params', n_params)
    setattr(module, 'n_nano_seconds', n_nano_seconds)
    FLAGS.last_module = module
    if isinstance(module, (SwitchableBatchNorm2d, SPBatchNorm2d)):
        if FLAGS.flops_mobile_mode:
            module.n_macs[idx] = 0
        else:
            module.n_macs[idx] = int(np.prod(list(ins)) * 2)
        module.n_params[idx] = get_params(module)
        if flops_and_params_only:
            module.n_nano_seconds[idx] = 0
        else:
            module.n_nano_seconds[idx] = run_forward(module, inputs)
        module.name = conv_module_name_filter(module.__repr__())
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        in_channels = ins[1]
        out_channels = outs[1]
        kernel_dims = list(module.kernel_size)
        groups = module.groups
        conv_per_position_flops = np.prod(kernel_dims) * in_channels * out_channels // groups

        batch_size = outs[0]
        output_dims = list(outs[2:])
        active_elements_count = batch_size * np.prod(output_dims)

        bias_flops = 0
        if module.bias is not None:
            bias_flops = out_channels
        overall_flops = (conv_per_position_flops + bias_flops) * active_elements_count
        module.n_macs[idx] = int(overall_flops)
        module.n_params[idx] = get_params(module)
        if flops_and_params_only:
            module.n_nano_seconds[idx] = 0
        else:
            module.n_nano_seconds[idx] = run_forward(module, inputs)
        module.name = conv_module_name_filter(module.__repr__())
    elif isinstance(module, nn.Linear):
        batch_size = outs[0]
        in_features = ins[1]
        out_features = outs[1]
        bias_macs = out_features if module.bias is not None else 0
        module.n_macs[idx] = int((in_features * out_features + bias_macs) * batch_size)
        module.n_params[idx] = get_params(module)
        if flops_and_params_only:
            module.n_nano_seconds[idx] = 0
        else:
            module.n_nano_seconds[idx] = run_forward(module, inputs)
        module.name = module.__repr__()
    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, SAvgPool2d)):
        if FLAGS.flops_mobile_mode:
            module.n_macs[idx] = 0
        else:
            module.n_macs[idx] = int(np.prod(list(ins)))
        module.n_params[idx] = 0
        if flops_and_params_only:
            module.n_nano_seconds[idx] = 0
        else:
            module.n_nano_seconds[idx] = run_forward(module, inputs)
        module.name = module.__repr__()
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU)):
        if FLAGS.flops_mobile_mode:
            module.n_macs[idx] = 0
        else:
            module.n_macs[idx] = int(np.prod(list(outs)))
        module.n_params[idx] = 0
        if flops_and_params_only:
            module.n_nano_seconds[idx] = 0
        else:
            module.n_nano_seconds[idx] = run_forward(module, inputs)
        module.name = module.__repr__()
    else:
        module.n_macs[idx] = 0
        module.n_params[idx] = 0
        module.n_nano_seconds[idx] = 0
        if type(module) not in ignore_zeros_t:
            for m in module.children():
                module.n_macs[idx] += m.n_macs[idx]
                module.n_params[idx] += m.n_params[idx]
                module.n_nano_seconds[idx] += m.n_nano_seconds[idx]
            if module.n_macs[idx] == 0:
                i = FLAGS.last_module.idx[0]
                j = FLAGS.last_module.idx[1]
                print(f'{i}.{j} of {module.width_mult}x: '
                      f'{type(module)} has zero n_macs.')
    return


def add_profiling_hooks(m, flops_and_params_only=False):

    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(
            lambda module, inputs, outputs: module_profiling(
                module, inputs, outputs,
                flops_and_params_only=flops_and_params_only
            )
        )
    )


def remove_profiling_hooks():

    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, batch=1, channel=3, use_cuda=True,
                    flops_and_params_only=False, no_print=False):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).
    """
    data = torch.rand(batch, channel, height, width)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = model.to(device)
    data = data.to(device)

    if not flops_and_params_only:
        model.eval()
    model.apply(lambda m: add_profiling_hooks(
        m, flops_and_params_only=flops_and_params_only
    ))

    with torch.no_grad():
        model(data)

    if getattr(model, 'width_mult', False) and not no_print:
        print(f'Model profiling with width mult {model.width_mult}x:')
    if not no_print:
        print(
            'params'.rjust(macs_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
    idx = FLAGS.idx
    if not no_print:
        print(
            f'{model.n_params[idx]:,}'.rjust(params_space, ' ') +
            f'{model.n_macs[idx]:,}'.rjust(macs_space, ' ') +
            f'{int(model.n_nano_seconds[idx]):,}'.rjust(seconds_space, ' '))
    remove_profiling_hooks()
    FLAGS.model_instance = model

    return model.n_macs[idx], model.n_params[idx]
