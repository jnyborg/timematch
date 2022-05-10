import torch
import argparse

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


def to_cuda(sample, device, non_blocking=True):
    pixels = sample['pixels'].cuda(device=device, non_blocking=non_blocking)
    valid_pixels = sample['valid_pixels'].cuda(device=device, non_blocking=non_blocking)
    positions = sample['positions'].cuda(device=device, non_blocking=non_blocking)
    if 'extra' in sample:
        extra = sample['extra'].cuda(device=device, non_blocking=non_blocking)
    else:
        extra = None
    return pixels, valid_pixels, positions, extra


def cat_samples(samples):
    out = {
        'pixels': torch.cat([x['pixels'] for x in samples]),
        'valid_pixels': torch.cat([x['valid_pixels'] for x in samples]),
        'positions': torch.cat([x['positions'] for x in samples]),
        'label': torch.cat([x['label'] for x in samples]),
    }
    if 'extra' in samples[0]:
        out['extra'] = torch.cat([x['extra'] for x in samples])
    return out


def onehot(label, num_classes, device='cpu'):
    return torch.zeros(label.size(0), num_classes, device=device).scatter_(1, label.view(-1, 1), 1)

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
