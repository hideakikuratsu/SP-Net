import builtins
import importlib
import math
import os
import random
import sys
import time
import warnings

import albumentations as A
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from pathlib import Path

from models.slimmable_ops import SConv2d, SPBatchNorm2d, SSqueezeExcitation
from models.slimmable_ops import bn_calibration_init
from utils.config import FLAGS
from utils.distributed import DistributedEvalSampler
from utils.distributed import master_only, wrap_model, get_bare_model, is_master
from utils.distributed import master_only_print as print
from utils.distributed import optimizer_to, scheduler_to, get_rank
from utils.meters import ScalarMeter, flush_scalar_meters
from utils.model_profiling import model_profiling

rng: np.random._generator.Generator


# noinspection PyUnresolvedReferences
def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)

    return model


def get_trans(size, mean, std, is_train, randaug=False):

    if is_train:
        if not randaug:
            trans = A.Compose([
                A.RandomResizedCrop(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            trans = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandAugment(FLAGS.randaug_N, FLAGS.randaug_M),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    else:
        if not randaug:
            trans = A.Compose([
                A.SmallestMaxSize(max_size=FLAGS.max_image_size),
                A.CenterCrop(height=size, width=size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize(FLAGS.max_image_size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    return trans


class AlbumentationsMNISTDataset(torchvision.datasets.MNIST):

    def __init__(self, root="~/data/mnist", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class AlbumentationsCIFAR10Dataset(torchvision.datasets.CIFAR10):

    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class AlbumentationsCIFAR100Dataset(torchvision.datasets.CIFAR100):

    def __init__(self, root="~/data/cifar100", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class AlbumentationsImageFolderDataset(ImageFolder):

    def __init__(self, root="~/ILSVRC2012_img_train", transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            transformed = self.transform(image=np.array(sample))
            sample = transformed['image']

        return sample, target


# noinspection PyTypeChecker
def get_datasets(dataset_name, use_aug=True):
    if dataset_name == 'mnist':
        mean, std = 0.5, 0.5
        transform_test = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=False)

        train_set = AlbumentationsMNISTDataset(
            root='~/pytorch_data', train=True, download=True, transform=transform_test
        )
        test_set = AlbumentationsMNISTDataset(
            root='~/pytorch_data', train=False, download=True, transform=transform_test
        )
        output_size = 10
    elif dataset_name == 'cifar10':
        mean = (0.491, 0.482, 0.447)
        std = (0.247, 0.243, 0.262)
        transform_train = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=use_aug)
        transform_test = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=False)

        train_set = AlbumentationsCIFAR10Dataset(
            root='~/pytorch_data', train=True, download=True, transform=transform_train
        )
        test_set = AlbumentationsCIFAR10Dataset(
            root='~/pytorch_data', train=False, download=True, transform=transform_test
        )
        output_size = 10
    elif dataset_name == 'cifar100':
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
        transform_train = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=use_aug)
        transform_test = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=False)

        train_set = AlbumentationsCIFAR100Dataset(
            root='~/pytorch_data', train=True, download=True, transform=transform_train
        )
        test_set = AlbumentationsCIFAR100Dataset(
            root='~/pytorch_data', train=False, download=True, transform=transform_test
        )
        output_size = 100
    elif dataset_name == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=use_aug)
        transform_test = get_trans(size=FLAGS.image_size, mean=mean, std=std, is_train=False)

        train_set = AlbumentationsImageFolderDataset(
            root='/home/hideaki/ILSVRC2012_img_train', transform=transform_train
        )
        test_set = AlbumentationsImageFolderDataset(
            root='/home/hideaki/ILSVRC2012_img_val_for_ImageFolder',
            transform=transform_test
        )
        output_size = 1000
    elif dataset_name == 'imagenet_randaug':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = get_trans(
            size=FLAGS.image_size, mean=mean, std=std, is_train=use_aug, randaug=True
        )
        transform_test = get_trans(
            size=FLAGS.image_size, mean=mean, std=std, is_train=False, randaug=True
        )

        train_set = ImageFolder(
            root='/home/hideaki/ILSVRC2012_img_train', transform=transform_train
        )
        test_set = ImageFolder(
            root='/home/hideaki/ILSVRC2012_img_val_for_ImageFolder',
            transform=transform_test
        )
        output_size = 1000
    else:
        sys.exit(0)

    FLAGS.data_size_train = len(train_set)
    FLAGS.data_size_test = len(test_set)
    FLAGS.num_classes = output_size

    return train_set, test_set


def get_loader(train_set, test_set, batch_size, test_batch_size, num_workers=6):

    if FLAGS.distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedEvalSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


class MyStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (float): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


def get_lr_scheduler(optimizer):
    """get learning rate"""
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma
        )
    elif FLAGS.lr_scheduler == 'step':
        lr_scheduler = MyStepLR(
            optimizer=optimizer, step_size=FLAGS.step_size,
            gamma=FLAGS.step_gamma
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def lr_schedule_per_iteration(train_loader, optimizer, scheduler, epoch, batch_idx=0):
    """ function for learning rate scheduling per iteration """
    epoch -= 1
    num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
    iters_per_epoch = len(train_loader)
    current_iter = epoch * iters_per_epoch + batch_idx + 1
    if FLAGS.warmup_epochs != 0 and epoch < FLAGS.warmup_epochs:
        linear_decaying_per_step = FLAGS.lr / FLAGS.warmup_epochs / iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_iter * linear_decaying_per_step
    elif scheduler is not None and FLAGS.warmup_epochs != 0 and epoch == FLAGS.warmup_epochs:
        scheduler.last_epoch = epoch
    elif FLAGS.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = FLAGS.lr / num_epochs / iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] -= linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        mult = (1. + math.cos(
            math.pi *
            (current_iter - FLAGS.warmup_epochs * iters_per_epoch) / num_epochs / iters_per_epoch
        )) / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = FLAGS.lr * mult
    else:
        pass


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay, nesterov=FLAGS.nesterov
        )
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay
        )
    elif FLAGS.optimizer == 'sgd_dw':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # `we found that it was important to put very little or
        # no weight decay (l2 regularization) on the depthwise filters
        # since their are so few parameters in them` from original paper
        model_params = []
        FLAGS.model_params = model_params
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] == 1:
                weight_decay = 0
            else:
                weight_decay = FLAGS.weight_decay
            item = {'params': params, 'weight_decay': weight_decay}
            model_params.append(item)
        optimizer = torch.optim.SGD(
            params=model_params, lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=FLAGS.nesterov
        )
    elif FLAGS.optimizer == 'sgd_no_bias_decay':
        # `As pointed out by Jia et al.
        # [Highly scalable deep learning training system with mixed-precision:
        # Training imagenet in four minutes.],
        # however, it’s recommended to only apply the regularization to
        # weights to avoid overfitting. The no bias decay heuristic
        # follows this recommendation, it only applies the weight decay to
        # the weights in convolution and fully-connected layers.
        # Other parameters, including the biases and γ and β in BN layers,
        # are left unregularized.` cited from He et al.
        # [Bag of Tricks for Image Classification with Convolutional Neural Networks]
        model_params = []
        FLAGS.model_params = model_params
        for params in model.parameters():
            ps = list(params.size())
            if (len(ps) == 4 and ps[1] != 1) or len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay}
            model_params.append(item)
        optimizer = torch.optim.SGD(
            params=model_params, lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=FLAGS.nesterov
        )
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                f'Optimizer {FLAGS.optimizer} is not yet implemented.')
    return optimizer


def set_random_seed():
    """set random seed"""
    global rng
    seed = FLAGS.random_seed + int(1e5 * get_rank())
    FLAGS.random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.cuda.manual_seed(seed)


def get_single_meter(phase):

    meter = {'loss': ScalarMeter()}
    for k in FLAGS.topk:
        meter[f'top{k}_error'] = ScalarMeter()
    if phase == 'train':
        meter['lr'] = ScalarMeter()
    return meter


def get_meters(phase):
    """util function for meters"""

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    if FLAGS.slimmable_training:
        meters = {}
        for width_mult in FLAGS.width_mult_list:
            meters[str(width_mult)] = get_single_meter(phase)
    else:
        meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter()

    return meters


# noinspection PyUnboundLocalVariable
@master_only
def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print(f'Start model profiling, use_cuda: {use_cuda}.')
    if FLAGS.slimmable_training:
        for width_mult in FLAGS.width_mult_list[::-1]:
            if FLAGS.train_only_target_width and FLAGS.target_width != width_mult:
                continue
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            FLAGS.idx = FLAGS.width_mult_list.index(width_mult)
            flops, params = model_profiling(model, FLAGS.image_size, FLAGS.image_size,
                                            batch=FLAGS.profiling_batch_size,
                                            use_cuda=use_cuda)
    else:
        model.apply(lambda m: setattr(m, 'width_mult', FLAGS.max_width))
        FLAGS.idx = FLAGS.width_mult_list.index(FLAGS.max_width)
        flops, params = model_profiling(model, FLAGS.image_size, FLAGS.image_size,
                                        batch=FLAGS.profiling_batch_size,
                                        use_cuda=use_cuda)
    return flops, params


# additional subgradient descent on the sparsity-induced penalty term
def update_BN(m):

    if isinstance(m, SPBatchNorm2d):
        m.weight.grad.data.add_(
            FLAGS.slimming_sparsity_rate * torch.sign(m.weight.data.clone())
        )


def mixup_criterion(criterion, outputs, targets):

    lam = FLAGS.mixup_lambda
    return lam * criterion(outputs, targets[0]) + (1 - lam) * criterion(outputs, targets[1])


def mixup_data(data, targets, alpha, generator):

    lam = np.random.beta(alpha, alpha)
    FLAGS.mixup_lambda = lam
    indices = torch.randperm(
        data.size(0), device=data.device, dtype=torch.long, generator=generator
    )
    mixed_data = lam * data + (1 - lam) * data[indices, ...]
    return mixed_data, [targets, targets[indices]]


def calculate_topk_error_sum(outputs, targets, k):

    _, pred = outputs.topk(k)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct.float().sum(0)

    return torch.sum(1 - correct_k).item()


def loss_fn_kd(hard_loss, outputs, teacher_outputs, T=1, alpha=0.9):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    kd_loss = -torch.sum(
        F.softmax(teacher_outputs / T, dim=1) * F.log_softmax(outputs / T, dim=1), dim=1
    )
    loss = (alpha * T * T) * kd_loss + (1.0 - alpha) * hard_loss

    return loss


def forward_loss(model, criterion, data, targets, meter, return_soft_targets=False, scaler=None,
                 teacher_outputs=None):
    """forward model and return loss"""
    with autocast(enabled=scaler is not None):
        outputs = model(data)
        if FLAGS.do_mixup:
            loss = mixup_criterion(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        if teacher_outputs is not None:
            loss = loss_fn_kd(
                loss, outputs, teacher_outputs, FLAGS.kd_T, FLAGS.kd_alpha
            )

    # cache to meter
    if meter is not None:
        for k in FLAGS.topk:
            if FLAGS.do_mixup:
                lam = FLAGS.mixup_lambda
                topk_error_sum = (lam * calculate_topk_error_sum(outputs, targets[0], k)
                                  + (1 - lam) * calculate_topk_error_sum(outputs, targets[1], k))
            else:
                topk_error_sum = calculate_topk_error_sum(outputs, targets, k)
            meter[f'top{k}_error'].cache_sum(topk_error_sum, data.size(0))
        meter['loss'].cache_sum(loss.sum().item(), data.size(0))

    if return_soft_targets:
        return loss.mean(), outputs
    return loss.mean()


# noinspection PyUnboundLocalVariable
def run_one_epoch(epoch, loader, model, criterion, optimizer, meters, phase='train', scheduler=None,
                  scaler=None, ema=None):
    """run one epoch for train/val/test/cal"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    is_train = phase == 'train'
    if is_train:
        model.train()
        random.seed(FLAGS.random_seed + epoch)
        np.random.seed(FLAGS.random_seed + epoch)
        generator = torch.Generator(device=FLAGS.gpu)
        generator.manual_seed(FLAGS.random_seed + epoch)
        FLAGS.do_mixup = FLAGS.mixup
    else:
        model.eval()
        if phase == 'cal':
            model.apply(bn_calibration_init)
        FLAGS.do_mixup = False

    with tqdm(
            loader, total=len(loader), desc=f'[Epoch {epoch}]',
            leave=False, disable=(not is_master())
    ) as pbar:
        for batch_idx, (data, targets) in enumerate(pbar):
            if phase == 'cal' and batch_idx == FLAGS.bn_cal_batch_num:
                break
            data = data.to(FLAGS.gpu, non_blocking=True)
            targets = targets.to(FLAGS.gpu, non_blocking=True)
            if FLAGS.do_mixup:
                data, targets = mixup_data(data, targets, FLAGS.mixup_alpha, generator)
            if is_train:
                # change learning rate if necessary
                lr_schedule_per_iteration(loader, optimizer, scheduler, epoch, batch_idx)
                optimizer.zero_grad()
                if FLAGS.slimmable_training:
                    # slimmable model (s-nets)
                    for i, width_mult in enumerate(FLAGS.width_mult_list[::-1]):
                        if FLAGS.train_only_target_width and FLAGS.target_width != width_mult:
                            continue
                        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                        FLAGS.idx = FLAGS.width_mult_list.index(width_mult)
                        meter = meters[str(width_mult)]
                        if width_mult == FLAGS.max_width:
                            loss, teacher_outputs = forward_loss(
                                model, criterion, data, targets, meter,
                                return_soft_targets=True, scaler=scaler
                            )
                        else:
                            if FLAGS.inplace_kd:
                                loss = forward_loss(
                                    model, criterion, data, targets, meter,
                                    scaler=scaler, teacher_outputs=teacher_outputs.detach()
                                )
                            else:
                                loss = forward_loss(
                                    model, criterion, data, targets, meter, scaler=scaler
                                )
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    loss = forward_loss(model, criterion, data, targets, meters, scaler=scaler)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if FLAGS.update_BN:
                    model.apply(update_BN)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if ema is not None:
                    ema.update()

                if FLAGS.slimmable_training:
                    for width_mult in FLAGS.width_mult_list[::-1]:
                        if FLAGS.train_only_target_width and FLAGS.target_width != width_mult:
                            continue
                        meter = meters[str(width_mult)]
                        meter['lr'].cache(optimizer.param_groups[0]['lr'])
                else:
                    meters['lr'].cache(optimizer.param_groups[0]['lr'])
            else:
                if FLAGS.slimmable_training:
                    for width_mult in FLAGS.width_mult_list[::-1]:
                        if FLAGS.train_only_target_width and FLAGS.target_width != width_mult:
                            continue
                        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                        FLAGS.idx = FLAGS.width_mult_list.index(width_mult)
                        meter = meters[str(width_mult)]
                        if ema is None:
                            forward_loss(model, criterion, data, targets, meter, scaler=scaler)
                        else:
                            with ema.average_parameters():
                                forward_loss(model, criterion, data, targets, meter, scaler=scaler)
                else:
                    if ema is None:
                        forward_loss(model, criterion, data, targets, meters, scaler=scaler)
                    else:
                        with ema.average_parameters():
                            forward_loss(model, criterion, data, targets, meters, scaler=scaler)

    time_str = f'{time.time() - t_start:.1f}s'
    if not hasattr(FLAGS, 'time_str_len'):
        FLAGS.time_str_len = len(time_str)

    if FLAGS.slimmable_training:
        for width_mult in FLAGS.width_mult_list[::-1]:
            if FLAGS.train_only_target_width and FLAGS.target_width != width_mult:
                continue
            results = flush_scalar_meters(meters[str(width_mult)])

            print(f'{time_str:>{FLAGS.time_str_len}}  {phase:>5}  '
                  f'{str(width_mult):>{FLAGS.width_str_len}}  '
                  f'{epoch:>{len(str(FLAGS.num_epochs))}}/{FLAGS.num_epochs}:  ' +
                  ', '.join(f'{k}: {v:.4f}' for k, v in results.items()))
    else:
        results = flush_scalar_meters(meters)

        print(f'{time_str:>{FLAGS.time_str_len}}  {phase:>5}  '
              f'{epoch:>{len(str(FLAGS.num_epochs))}}/{FLAGS.num_epochs}:  ' +
              ', '.join(f'{k}: {v:.4f}' for k, v in results.items()))

    return results['top1_error']


def set_m_list(m_list, m_list_width_num, ms, reverse=False, get_start_masks=False, is_conv=False):

    start_idx = 0 if not reverse else -1
    if not get_start_masks:
        next_m = ms[start_idx]
        if isinstance(next_m, list):
            next_m = next_m[start_idx]
        while next_m is not None:
            m_list.append(next_m)
            if not reverse:
                next_m = next_m.next[0]
            else:
                next_m = next_m.prev[0]

    next_m = ms[start_idx]
    if isinstance(next_m, list):
        next_m = next_m[start_idx]

    for width_mult in FLAGS.width_mult_list:
        m_list_width_num[str(width_mult)] = []
    while next_m is not None:
        for width_idx, width_mult in enumerate(FLAGS.width_mult_list):
            if get_start_masks:
                if is_conv:
                    if next_m.prev[0] is not None:
                        features_list = next_m.prev[0].out_channels_list
                    else:
                        features_list = next_m.in_channels_list
                else:
                    raise ValueError(f'Combination of get_start_masks = True and '
                                     f'is_conv = False not be allowed')
            else:
                if is_conv:
                    features_list = next_m.out_channels_list
                else:
                    features_list = next_m.num_features_list
            features = features_list[width_idx]
            m_list_width_num[str(width_mult)].append(features)
        if not reverse:
            next_m = next_m.next[0]
        else:
            next_m = next_m.prev[0]


def prune_model(base_model_path, base_model, org_model, multi_base_pruning=False, offset=0):

    # If you include 'running' params, size mismatch error occurs because sizes of
    # BN running params on base network are different from ones on newly moulded model, and
    # BN modules will be replaced by pruned ones afterwards anyway. strict=False is needed
    # because running params are not to be transferred (missing params error occurs).
    if FLAGS.set_pretrained_base:
        org_checkpoint = torch.load(FLAGS.base_model_path, map_location='cpu')
        org_model.load_state_dict({k: v for k, v in org_checkpoint['model'].items()
                                   if 'running' not in k}, strict=False)
    checkpoint = torch.load(base_model_path, map_location='cpu')
    base_model.load_state_dict({k: v for k, v in checkpoint['model'].items()
                                if 'running' not in k}, strict=False)
    print(f'Loaded model {base_model_path}.')
    print(f'Best val: {checkpoint["best_val"]:.4f}')

    total = 0
    for m in base_model.modules():
        if isinstance(m, SPBatchNorm2d):
            total += m.weight.data.size(0)
    FLAGS.total = total

    m_vals = torch.zeros(total)
    index = 0
    for m in base_model.modules():
        if FLAGS.l1_based_pruning:
            if isinstance(m, SConv2d):
                size = m.weight.data.size(0)
                m_vals[index:(index + size)] = m.weight.data.clone().abs().sum(dim=(1, 2, 3))
                index += size
        else:
            if isinstance(m, SPBatchNorm2d):
                size = m.weight.data.size(0)
                m_vals[index:(index + size)] = m.weight.data.clone().abs()
                index += size

    y, _ = torch.sort(m_vals)
    thresholds = []
    for width_mult in FLAGS.width_mult_list[:-1]:
        threshold_index = int(total * (1 - width_mult))
        thresholds.append(y[threshold_index])
    FLAGS.thresholds = thresholds

    base_conv_ms = FLAGS.conv_ms
    base_bn_ms = FLAGS.bn_ms
    if multi_base_pruning:
        org_conv_ms = FLAGS.org_conv_ms
        org_bn_ms = FLAGS.org_bn_ms
    else:
        org_conv_ms = base_conv_ms
        org_bn_ms = base_bn_ms
    for i, n_bn in enumerate(FLAGS.cfg_only_bn):
        if isinstance(n_bn, list):
            for j, v in enumerate(n_bn):
                if type(v) is not int:
                    break
                org_conv_m = org_conv_ms[i][j]
                org_bn_m = org_bn_ms[i][j]
                base_conv_m = base_conv_ms[i][j]
                base_bn_m = base_bn_ms[i][j]
                need_mask = getattr(org_conv_m, 'need_mask', False)
                end_of_block = getattr(org_conv_m, 'end_of_block', False)
                shortcut = getattr(org_conv_m, 'shortcut', False)

                end_mask_sizes = []
                if offset == 0 and (need_mask or end_of_block or shortcut or FLAGS.mask_pruning):
                    org_conv_m.masks = []
                    if end_of_block:
                        org_conv_m.fused_masks = []
                # noinspection PyUnboundLocalVariable
                for width_idx, threshold in enumerate(thresholds):
                    base_m_weight_abs = torch.zeros_like(org_bn_m.weight.data)
                    if FLAGS.l1_based_pruning:
                        base_m_weight_abs[:base_bn_m.weight.data.size(0)] = \
                            base_conv_m.weight.data.clone().abs().sum(dim=(1, 2, 3))
                    else:
                        base_m_weight_abs[:base_bn_m.weight.data.size(0)] = \
                            base_bn_m.weight.data.clone().abs()
                    if base_conv_m.groups_list == [1] * len(base_conv_m.groups_list):
                        end_mask_size = base_m_weight_abs.gt(threshold).sum().item()
                    else:
                        end_mask_size = base_conv_m.prev[0].out_channels_list[width_idx]
                    max_end_mask_size = v
                    if FLAGS.zpm_pruning:
                        if end_of_block or shortcut:
                            min_size = int(max_end_mask_size * FLAGS.min_size_ratio / 2)
                        else:
                            min_size = int(max_end_mask_size * FLAGS.min_size_ratio)
                    else:
                        min_size = int(max_end_mask_size * FLAGS.min_size_ratio)
                    if end_mask_size < min_size:
                        if not FLAGS.profiling_only or FLAGS.profiling_verbose:
                            print(f'change mask size: {end_mask_size} ---> {min_size}')
                        end_mask_size = min_size
                    if need_mask or end_of_block or shortcut or FLAGS.mask_pruning:
                        if FLAGS.set_pretrained_base:
                            if FLAGS.l1_based_pruning:
                                org_m_weight_abs = \
                                    org_conv_m.weight.data.clone().abs().sum(dim=(1, 2, 3))
                            else:
                                org_m_weight_abs = org_bn_m.weight.data.clone().abs()
                            _, topk_idx = torch.sort(org_m_weight_abs, descending=True)
                        else:
                            _, topk_idx = torch.sort(base_m_weight_abs, descending=True)
                        mask = torch.zeros_like(base_m_weight_abs).int()
                        mask[topk_idx[:end_mask_size]] = 1
                        org_conv_m.masks.append(mask)
                        if end_of_block and n_bn[-1]:
                            prev_block = org_conv_ms[i - 1]
                            idx = -1 if (getattr(prev_block[-1], 'end_of_block', False) or
                                         getattr(prev_block[-1], 'need_mask', False)) else -2
                            prev_last_conv_m = prev_block[idx]
                            if getattr(prev_last_conv_m, 'need_mask', False):
                                prev_mask = prev_last_conv_m.masks[width_idx + offset]
                            elif getattr(prev_last_conv_m, 'end_of_block', False):
                                prev_mask = prev_last_conv_m.fused_masks[width_idx + offset]
                            else:
                                raise ValueError('prev_last_conv_m should have either '
                                                 '"need_mask" or "end_of_block" attr')
                            org_conv_m.fused_masks.append(prev_mask | mask)
                            if FLAGS.mask_pruning:
                                org_conv_m.masks[-1] = prev_block[idx].masks[width_idx + offset]
                        elif shortcut:
                            block_mask = org_conv_ms[i][-2].masks[width_idx + offset]
                            org_conv_ms[i][-2].fused_masks.append(block_mask | mask)
                            if FLAGS.mask_pruning:
                                org_conv_ms[i][-2].masks[width_idx + offset] = mask
                    end_mask_sizes.append(end_mask_size)

                    if not FLAGS.profiling_only or FLAGS.profiling_verbose:
                        print(f'[{FLAGS.width_mult_list[width_idx]:<0{FLAGS.width_str_len}}] '
                              f'cfg index: {i:d}.{j} \t '
                              f'remain/total: {end_mask_size:d}/{max_end_mask_size:d}')

                if base_conv_m.groups_list == [1] * len(base_conv_m.groups_list):
                    if FLAGS.zpm_pruning:
                        base_conv_m.out_channels_list[:len(end_mask_sizes)] = end_mask_sizes
                        base_bn_m.num_features_list[:len(end_mask_sizes)] = end_mask_sizes
                        org_conv_m.out_channels_list[offset:offset + len(end_mask_sizes)] = \
                            end_mask_sizes
                        org_bn_m.num_features_list[offset:offset + len(end_mask_sizes)] = \
                            end_mask_sizes
                    elif not end_of_block:
                        base_conv_m.out_channels_list[:len(end_mask_sizes)] = end_mask_sizes
                        base_bn_m.num_features_list[:len(end_mask_sizes)] = end_mask_sizes

                if FLAGS.mask_pruning:
                    for k, num_features in enumerate(base_bn_m.num_features_list[:-1]):
                        org_bn_m.bn[k] = nn.BatchNorm2d(num_features, affine=False)
                        if shortcut:
                            org_bn_ms[i][-2].bn[k] = nn.BatchNorm2d(num_features, affine=False)
        elif isinstance(n_bn, int):
            org_conv_m = org_conv_ms[i]
            org_bn_m = org_bn_ms[i]
            base_conv_m = base_conv_ms[i]
            base_bn_m = base_bn_ms[i]
            if getattr(org_conv_m, 'no_prune', False):
                if not FLAGS.profiling_only or FLAGS.profiling_verbose:
                    for width_idx in range(len(thresholds)):
                        max_end_mask_size = org_conv_m.out_channels_list[width_idx + offset]
                        print(f'[{FLAGS.width_mult_list[width_idx]:<0{FLAGS.width_str_len}}] '
                              f'cfg index: {i:d} \t '
                              f'remain/total: {max_end_mask_size:d}/{max_end_mask_size:d}')
                continue

            end_mask_sizes = []
            if FLAGS.mask_pruning:
                org_conv_m.masks = []
            for width_idx, threshold in enumerate(thresholds):
                base_m_weight_abs = torch.zeros_like(org_bn_m.weight.data)
                if FLAGS.l1_based_pruning:
                    base_m_weight_abs[:base_bn_m.weight.data.size(0)] = \
                        base_conv_m.weight.data.clone().abs().sum(dim=(1, 2, 3))
                else:
                    base_m_weight_abs[:base_bn_m.weight.data.size(0)] = \
                        base_bn_m.weight.data.clone().abs()
                if base_conv_m.groups_list == [1] * len(base_conv_m.groups_list):
                    end_mask_size = base_m_weight_abs.gt(threshold).sum().item()
                else:
                    end_mask_size = base_conv_m.prev[0].out_channels_list[width_idx]
                max_end_mask_size = n_bn
                min_size = int(max_end_mask_size * FLAGS.min_size_ratio)
                if end_mask_size < min_size:
                    if not FLAGS.profiling_only or FLAGS.profiling_verbose:
                        print(f'change mask size: {end_mask_size} ---> {min_size}')
                    end_mask_size = min_size
                if FLAGS.mask_pruning:
                    mask = base_m_weight_abs.gt(threshold).int()
                    org_conv_m.masks.append(mask)
                end_mask_sizes.append(end_mask_size)

                if not FLAGS.profiling_only or FLAGS.profiling_verbose:
                    print(f'[{FLAGS.width_mult_list[width_idx]:<0{FLAGS.width_str_len}}] '
                          f'cfg index: {i:d} \t '
                          f'remain/total: {end_mask_size:d}/{max_end_mask_size:d}')

            if base_conv_m.groups_list == [1] * len(base_conv_m.groups_list):
                base_conv_m.out_channels_list[:len(end_mask_sizes)] = end_mask_sizes
                base_bn_m.num_features_list[:len(end_mask_sizes)] = end_mask_sizes
                org_conv_m.out_channels_list[offset:offset + len(end_mask_sizes)] = end_mask_sizes
                org_bn_m.num_features_list[offset:offset + len(end_mask_sizes)] = end_mask_sizes

            if FLAGS.mask_pruning:
                for k, num_features in enumerate(base_bn_m.num_features_list[:-1]):
                    org_bn_m.bn[k] = nn.BatchNorm2d(num_features, affine=False)
        else:
            raise ValueError(f'Not correct type: {type(n_bn)} for n_bn')

    if 'efficientnet' in FLAGS.model:
        for m in org_model.modules():
            if type(m) is SSqueezeExcitation:
                prev_conv_m = m.prev[0]
                prev_last_conv_m = org_conv_ms[prev_conv_m.idx[0] - 1]
                if type(prev_last_conv_m) is list:
                    prev_last_conv_m = prev_last_conv_m[-1]
                if getattr(prev_last_conv_m, 'end_of_block', False):
                    for i, fused_mask in enumerate(prev_last_conv_m.fused_masks):
                        m.fc1.out_channels_list[i] = max(1, fused_mask.count_nonzero().item() // 4)
                else:
                    m.fc1.out_channels_list = [max(1, i // 4) for i in
                                               prev_last_conv_m.out_channels_list]
                m.fc2.out_channels_list = prev_conv_m.out_channels_list.copy()

    FLAGS.m_list = []
    FLAGS.m_list_reverse = []
    FLAGS.m_list_bn_num = {}
    FLAGS.m_list_bn_num_reverse = {}
    set_m_list(m_list=FLAGS.m_list, m_list_width_num=FLAGS.m_list_bn_num,
               ms=base_bn_ms, reverse=False)
    set_m_list(m_list=FLAGS.m_list_reverse, m_list_width_num=FLAGS.m_list_bn_num_reverse,
               ms=base_bn_ms, reverse=True)
    FLAGS.conv_m_list = []
    FLAGS.conv_m_list_reverse = []
    FLAGS.conv_m_list_ch_num = {}
    FLAGS.conv_m_list_ch_num_reverse = {}
    FLAGS.conv_m_list_start_ch_num = {}
    set_m_list(m_list=FLAGS.conv_m_list, m_list_width_num=FLAGS.conv_m_list_ch_num, ms=base_conv_ms,
               reverse=False, is_conv=True)
    set_m_list(m_list=FLAGS.conv_m_list_reverse,
               m_list_width_num=FLAGS.conv_m_list_ch_num_reverse, ms=base_conv_ms, reverse=True,
               is_conv=True)
    set_m_list(m_list=FLAGS.conv_m_list, m_list_width_num=FLAGS.conv_m_list_start_ch_num,
               ms=base_conv_ms, reverse=False, get_start_masks=True, is_conv=True)

    if not FLAGS.profiling_only or FLAGS.profiling_verbose:
        print('remain ratio')
        for k, v in FLAGS.m_list_bn_num.items():
            print(f'{k:<0{FLAGS.width_str_len}}x: '
                  f'{sum(v):>{len(str(total))}}/{total} = {sum(v) / total * 100:.4f}%')

        print('BN features num order')
        for k, v in FLAGS.m_list_bn_num.items():
            print(f'{k:<0{FLAGS.width_str_len}}x: {v}')

        print('BN features num reverse order')
        for k, v in FLAGS.m_list_bn_num_reverse.items():
            print(f'{k:<0{FLAGS.width_str_len}}x: {v}')

    for v, v2 in zip(FLAGS.m_list_bn_num.values(), FLAGS.conv_m_list_ch_num.values()):
        assert v == v2

    for v, v2 in zip(FLAGS.m_list_bn_num_reverse.values(),
                     FLAGS.conv_m_list_ch_num_reverse.values()):
        assert v == v2

    if not FLAGS.profiling_only or FLAGS.profiling_verbose:
        print('BN end_mask num == Conv end_mask num: OK')

    if not multi_base_pruning or (
            multi_base_pruning and offset == len(FLAGS.base_width_mult_list) - 1
    ):
        if FLAGS.reinit_params:
            org_model.reset_parameters(init_bn_val=FLAGS.init_bn_val, reinit=True)
        elif FLAGS.mask_pruning:
            pass
        else:
            sort_model_weight_by_slim_bn(model=org_model, multi_base_pruning=multi_base_pruning)

    print('Pre-processing Successful!')

    return base_model


def sort_model_weight_by_slim_bn(model, multi_base_pruning=False):

    if multi_base_pruning:
        conv_ms = FLAGS.org_conv_ms
        bn_ms = FLAGS.org_bn_ms
    else:
        conv_ms = FLAGS.conv_ms
        bn_ms = FLAGS.bn_ms
    for i, n_bn in enumerate(FLAGS.cfg_only_bn):
        if isinstance(n_bn, list):
            for j, v in enumerate(n_bn):
                if type(v) is not int:
                    break
                conv_m = conv_ms[i][j]
                bn_m = bn_ms[i][j]
                need_mask = getattr(conv_m, 'need_mask', False)
                end_of_block = getattr(conv_m, 'end_of_block', False)
                shortcut = getattr(conv_m, 'shortcut', False)

                if FLAGS.l1_based_pruning:
                    m_weight_abs = conv_m.weight.data.clone().abs().sum(dim=(1, 2, 3))
                else:
                    m_weight_abs = bn_m.weight.data.clone().abs()
                _, end_sort_idx = torch.sort(m_weight_abs, descending=True)

                if conv_m.groups_list == [1] * len(conv_m.groups_list):
                    conv_m.end_sort_idx = end_sort_idx
                    bn_m.end_sort_idx = end_sort_idx
                else:
                    conv_m.end_sort_idx = conv_m.prev[0].end_sort_idx
                    bn_m.end_sort_idx = bn_m.prev[0].end_sort_idx

                if FLAGS.zpm_pruning:
                    if need_mask or end_of_block or shortcut:
                        restore_sort_idx_list = []
                        for out_channels in conv_m.out_channels_list:
                            restore_sort_idx_list.append(torch.sort(end_sort_idx[:out_channels])[1])
                        conv_m.restore_sort_idx_list = restore_sort_idx_list
                else:
                    if end_of_block and n_bn[-1]:
                        conv_m.end_sort_idx = conv_ms[i - 1][-1].end_sort_idx
                        bn_m.end_sort_idx = bn_ms[i - 1][-1].end_sort_idx
                    elif shortcut:
                        conv_ms[i][-2].end_sort_idx = end_sort_idx
                        bn_ms[i][-2].end_sort_idx = end_sort_idx
        elif isinstance(n_bn, int):
            conv_m = conv_ms[i]
            bn_m = bn_ms[i]

            if FLAGS.l1_based_pruning:
                m_weight_abs = conv_m.weight.data.clone().abs().sum(dim=(1, 2, 3))
            else:
                m_weight_abs = bn_m.weight.data.clone().abs()
            _, end_sort_idx = torch.sort(m_weight_abs, descending=True)

            if conv_m.groups_list == [1] * len(conv_m.groups_list):
                conv_m.end_sort_idx = end_sort_idx
                bn_m.end_sort_idx = end_sort_idx
            else:
                conv_m.end_sort_idx = conv_m.prev[0].end_sort_idx
                bn_m.end_sort_idx = bn_m.prev[0].end_sort_idx
        else:
            raise ValueError(f'Not correct type: {type(n_bn)} for n_bn')

    for m in model.modules():
        if isinstance(m, SConv2d):
            weight = m.weight.data.clone()

            if m.prev[0] is not None and m.groups_list == [1] * len(m.groups_list):
                if FLAGS.zpm_pruning:
                    if not (getattr(m.prev[0], 'end_of_block', False) or
                            getattr(m.prev[0], 'shortcut', False)):
                        start_sort_idx = m.prev[0].end_sort_idx
                        weight = weight[:, start_sort_idx, ...]
                else:
                    start_sort_idx = m.prev[0].end_sort_idx
                    weight = weight[:, start_sort_idx, ...]

            m.weight.data = weight[m.end_sort_idx, ...]
            if m.bias is not None:
                m.bias.data = m.bias.data.clone()[m.end_sort_idx]
        elif isinstance(m, SPBatchNorm2d):
            m.weight.data = m.weight.data.clone()[m.end_sort_idx]
            m.bias.data = m.bias.data.clone()[m.end_sort_idx]

            m.bn = nn.ModuleList(
                [nn.BatchNorm2d(i, affine=False) for i in m.num_features_list[:-1]]
            )
        elif isinstance(m, nn.Linear):
            weight = m.weight.data.clone()

            if not (FLAGS.zpm_pruning and 'resnet' in FLAGS.model):
                m.weight.data = weight[:, m.prev[0].end_sort_idx]

    if 'efficientnet' in FLAGS.model:
        for m in model.modules():
            if type(m) is SSqueezeExcitation:
                prev_conv_m = m.prev[0]
                for i, se_conv_m in enumerate([m.fc1, m.fc2]):
                    weight = se_conv_m.weight.data.clone()
                    bias = se_conv_m.bias.data.clone()

                    end_sort_idx = prev_conv_m.end_sort_idx
                    if i == 0:
                        se_conv_m.weight.data = weight[:, end_sort_idx, ...]
                    else:
                        se_conv_m.weight.data = weight[end_sort_idx, ...]
                        se_conv_m.bias.data = bias[end_sort_idx]


def eliminate_width(model):

    idx = FLAGS.width_mult_list.index(FLAGS.eliminate_target_width) - 1
    assert idx >= 0, 'you should choose wider width for eliminate_target_width'
    for m in model.modules():
        if isinstance(m, SConv2d):
            if m.prev[0] is not None:
                in_channels = m.prev[0].out_channels_list[idx]
            else:
                in_channels = m.in_channels_list[idx]
            out_channels = m.out_channels_list[idx]
            m.weight.data = m.weight.data.clone()[:out_channels, :in_channels, ...]
            if m.bias is not None:
                m.bias.data = m.bias.data.clone()[:out_channels]

            m.in_channels_list = m.in_channels_list[:idx + 1]
            m.out_channels_list = m.out_channels_list[:idx + 1]
            m.groups_list = m.groups_list[:idx + 1]
        elif isinstance(m, SPBatchNorm2d):
            num_features = m.num_features_list[idx]
            m.weight.data = m.weight.data.clone()[:num_features]
            m.bias.data = m.bias.data.clone()[:num_features]

            m.num_features_list = m.num_features_list[:idx + 1]

            m.running_mean = m.bn[idx].running_mean
            m.running_var = m.bn[idx].running_var
            m.bn = m.bn[:idx]
        elif isinstance(m, nn.Linear):
            in_features = m.prev[0].num_features_list[idx]
            m.weight.data = m.weight.data.clone()[:, :in_features]

            m.in_features_list = m.in_features_list[:idx + 1]
            m.out_features_list = m.out_features_list[:idx + 1]

    if 'efficientnet' in FLAGS.model:
        for m in model.modules():
            if type(m) is SSqueezeExcitation:
                prev_conv_m = m.prev[0]
                for i, se_conv_m in enumerate([m.fc1, m.fc2]):
                    if i == 0:
                        in_channels = prev_conv_m.out_channels_list[idx]
                    else:
                        in_channels = m.fc1.out_channels_list[idx]
                    out_channels = se_conv_m.out_channels_list[idx]
                    se_conv_m.weight.data = se_conv_m.weight.data[:out_channels, :in_channels, ...]
                    se_conv_m.bias.data = se_conv_m.bias.data[:out_channels]

                    se_conv_m.in_channels_list = se_conv_m.in_channels_list[:idx + 1]
                    se_conv_m.out_channels_list = se_conv_m.out_channels_list[:idx + 1]

    FLAGS.width_mult_list = FLAGS.width_mult_list[:idx + 1]
    FLAGS.max_width = FLAGS.width_mult_list[-1]
    model.apply(lambda m: setattr(m, 'width_mult', FLAGS.max_width))
    FLAGS.idx = FLAGS.width_mult_list.index(FLAGS.max_width)


# noinspection PyUnresolvedReferences
def train_val_test(gpu, ngpus_per_node):
    """train and val"""
    if FLAGS.dist_url == 'env://' and not hasattr(FLAGS, 'world_size'):
        FLAGS.world_size = int(os.environ['WORLD_SIZE'])
    FLAGS.distributed = FLAGS.world_size > 1 or FLAGS.mp_dist
    FLAGS.gpu = gpu
    if FLAGS.gpu is not None:
        builtins.print(f'Use GPU: {FLAGS.gpu} for training')

    if FLAGS.distributed:
        if FLAGS.dist_url == 'env://' and not hasattr(FLAGS, 'rank'):
            FLAGS.rank = int(os.environ['RANK'])
        if FLAGS.mp_dist:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            FLAGS.world_size = ngpus_per_node * FLAGS.world_size
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            FLAGS.rank = FLAGS.rank * ngpus_per_node + FLAGS.gpu
        dist.init_process_group(backend=FLAGS.dist_backend, init_method=FLAGS.dist_url,
                                world_size=FLAGS.world_size, rank=FLAGS.rank)

    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = True
    # seed
    set_random_seed()

    FLAGS.max_width = FLAGS.width_mult_list[-1]
    FLAGS.min_width = FLAGS.width_mult_list[0]
    FLAGS.width_str_len = max([len(str(s)) for s in FLAGS.width_mult_list])

    # model
    if FLAGS.AMP:
        scaler = GradScaler()
    else:
        scaler = None
    train_set, test_set = get_datasets(dataset_name=FLAGS.dataset, use_aug=True)
    model = get_model()
    model.apply(lambda m: setattr(m, 'width_mult', FLAGS.max_width))
    FLAGS.idx = FLAGS.width_mult_list.index(FLAGS.max_width)
    criterion = nn.CrossEntropyLoss(label_smoothing=FLAGS.label_smoothing, reduction='none')

    if hasattr(FLAGS, 'base_model_path'):
        if FLAGS.multi_base_pruning:
            org_width_mult_list = FLAGS.width_mult_list.copy()
            FLAGS.org_conv_ms = FLAGS.conv_ms
            FLAGS.org_bn_ms = FLAGS.bn_ms
            for width_idx, (base_width, base_model_path) in \
                    enumerate(zip(FLAGS.base_width_mult_list, FLAGS.multi_base_model_path)):
                FLAGS.width_mult_list = [org_width_mult_list[width_idx], base_width]
                base_model = get_model()
                prune_model(
                    base_model_path, base_model, model, multi_base_pruning=True, offset=width_idx
                )
            FLAGS.width_mult_list = org_width_mult_list.copy()
            FLAGS.conv_ms = FLAGS.org_conv_ms
            FLAGS.bn_ms = FLAGS.org_bn_ms
        else:
            base_model_path = FLAGS.base_model_path
            model = prune_model(base_model_path, model, model)

    optimizer = get_optimizer(model)
    if FLAGS.enable_EMA:
        ema = ExponentialMovingAverage(model.parameters(), decay=FLAGS.EMA_decay)
    else:
        ema = None

    # check resume training
    if FLAGS.resume:
        log_dir = FLAGS.log_dir
        if hasattr(FLAGS, 'base_model_path'):
            log_dir = FLAGS.finetune_log_dir
        checkpoint_path = Path(log_dir, FLAGS.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError('Saved file for resume not exists!')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if FLAGS.enable_EMA:
            ema.load_state_dict(checkpoint['ema'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        if lr_scheduler is not None:
            lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
        print(
            f'Loaded checkpoint {checkpoint_path} at epoch {last_epoch}. Best val: {best_val:.4f}'
        )
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = 0
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        if FLAGS.profiling and not FLAGS.eliminate_width:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)
            if FLAGS.profiling_only:
                return
            dist.barrier()

    # check pretrained
    if FLAGS.pretrained:
        pretrained_path = FLAGS.pretrained
        if not Path(pretrained_path).exists():
            raise FileNotFoundError('Pretrained file not exists!')
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        best_val = checkpoint['best_val']
        print(f'Loaded best model {pretrained_path}. Best val: {best_val:.4f}')

    if FLAGS.eliminate_width:
        eliminate_width(model)
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)
            if FLAGS.profiling_only:
                return
            dist.barrier()

    model = wrap_model(model, ngpus_per_node)
    bare_model = get_bare_model(model)
    optimizer_to(optimizer, FLAGS.gpu)
    if lr_scheduler is not None:
        scheduler_to(lr_scheduler, FLAGS.gpu)

    # data
    train_loader, test_loader = get_loader(
        train_set, test_set, FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.num_workers
    )

    if FLAGS.test_only and test_loader is not None:
        print('Start testing.')
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(-1, test_loader, model, criterion, optimizer, test_meters,
                          phase='test', scaler=scaler, ema=ema)
        return

    print('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs + 1):
        if FLAGS.skip_training:
            print(f'Skip training at epoch: {epoch}')
            break

        if FLAGS.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train
        run_one_epoch(epoch, train_loader, model, criterion, optimizer, train_meters,
                      phase='train', scheduler=lr_scheduler, scaler=scaler, ema=ema)
        if lr_scheduler is not None:
            if FLAGS.warmup_epochs == 0 or \
                    (FLAGS.warmup_epochs != 0 and FLAGS.warmup_epochs < epoch):
                lr_scheduler.step()

        # val
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            # noinspection PyTypeChecker
            top1_err = run_one_epoch(epoch, test_loader, model, criterion, optimizer,
                                     val_meters, phase='val', scaler=scaler, ema=ema)
        # save latest and best checkpoint
        if is_master():
            save_dir = Path(FLAGS.finetune_log_dir if hasattr(FLAGS, 'base_model_path') else
                            FLAGS.log_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            # save best model
            if top1_err < best_val:
                best_val = top1_err
                save_path = save_dir / FLAGS.best_model_path
                torch.save(
                    {
                        'model': bare_model.state_dict(),
                        'ema': ema.state_dict() if FLAGS.enable_EMA else None,
                        'best_val': best_val,
                    }, save_path
                )
                print(f'New best validation top1 error: {best_val:.4f}')

            # save checkpoint
            save_path = save_dir / FLAGS.checkpoint_path
            torch.save(
                {
                    'model': bare_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict() if FLAGS.enable_EMA else None,
                    'last_epoch': epoch,
                    'best_val': best_val,
                    'meters': (train_meters, val_meters),
                }, save_path
            )

    if FLAGS.calibrate_bn:
        cal_meters = get_meters('cal')
        print('Start calibration.')
        run_one_epoch(-1, train_loader, model, criterion, optimizer, cal_meters,
                      phase='cal', scaler=scaler)
        print('Start validation after calibration.')
        with torch.no_grad():
            # noinspection PyTypeChecker
            run_one_epoch(-1, test_loader, model, criterion, optimizer, cal_meters,
                          phase='val', scaler=scaler)
        if is_master():
            best_model_dir = Path(FLAGS.log_dir)
            if not best_model_dir.exists():
                best_model_dir.mkdir(parents=True)
            best_model_path = best_model_dir / (FLAGS.best_model_path[:-3] + '_bn_calibrated.pt')
            torch.save(
                {
                    'model': bare_model.state_dict(),
                }, best_model_path
            )

    return


def main():
    """train and eval model"""
    ngpus_per_node = torch.cuda.device_count()
    if FLAGS.mp_dist:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train_val_test, nprocs=ngpus_per_node, args=(ngpus_per_node,))
    else:
        # Simply call main_worker function
        train_val_test(FLAGS.gpu, ngpus_per_node)


if __name__ == "__main__":
    main()
