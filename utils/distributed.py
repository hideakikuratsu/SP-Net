import functools

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Sampler

from utils.config import FLAGS


def get_bare_model(model):

    if type(model) in [nn.DataParallel, nn.parallel.DistributedDataParallel]:
        bare_model = model.module
    else:
        bare_model = model
        master_only_print('use original model as bare_model')

    return bare_model


def optimizer_to(optimizer, device):

    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(scheduler, device):

    for param in scheduler.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


def wrap_model(model, ngpus_per_node):

    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        FLAGS.gpu = 'cpu'
        print('using CPU, this will be slow')
    elif FLAGS.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if FLAGS.gpu is not None:
            torch.cuda.set_device(FLAGS.gpu)
            model = model.to(FLAGS.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            FLAGS.batch_size = int(FLAGS.batch_size / ngpus_per_node)
            FLAGS.test_batch_size = int(FLAGS.test_batch_size / ngpus_per_node)
            FLAGS.num_workers = int((FLAGS.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[FLAGS.gpu],
                find_unused_parameters=False if FLAGS.sp_model else True
            )
        else:
            FLAGS.gpu = device
            model = model.to(FLAGS.gpu)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=False if FLAGS.sp_model else True
            )
    elif FLAGS.gpu is not None:
        torch.cuda.set_device(FLAGS.gpu)
        model = model.to(FLAGS.gpu)
    else:
        FLAGS.gpu = device
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).to(FLAGS.gpu)

    return model


def get_rank():

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():

    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    """check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args, **kwargs):
    """master-only print"""
    print(*args, **kwargs)


def dist_reduce_tensor(tensor):
    """ Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if get_rank() == 0:
            tensor /= world_size
    return tensor


def dist_all_reduce_tensor(tensor):
    """ Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval'
                f' [0, {num_replicas - 1}]'
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # true value without extra samples
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # true value without extra samples
        self.num_samples = len(indices)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
