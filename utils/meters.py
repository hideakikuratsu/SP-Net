import torch
import torch.distributed as dist
from utils.config import FLAGS


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.
    """
    def __init__(self):

        self.sum = 0
        self.count = 0

    def reset(self):

        self.sum = 0
        self.count = 0

    def cache(self, value):

        self.sum += value
        self.count += 1

    def cache_sum(self, sum_val, count):

        self.sum += sum_val
        self.count += count

    def all_reduce(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()


class ScalarMeter(Meter):
    """ScalarMeter records scalar over steps.

    """
    def __init__(self):

        super().__init__()


def flush_scalar_meters(meters, method='avg'):

    results = {}
    assert isinstance(meters, dict), 'meters should be a dict.'
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            continue
        if FLAGS.distributed:
            meter.all_reduce()
        if method == 'avg':
            value = meter.sum / meter.count
        elif method == 'sum':
            value = meter.sum
        else:
            raise NotImplementedError(f'flush method: {method} is not yet implemented.')
        results[name] = value
        meter.reset()

    return results
