import random

import numpy as np
import torch


class Identity(object):
    def __call__(self, sample):
        return sample


class RandomSamplePixels(object):
    """Randomly draw num_pixels from the available pixels in sample.
    If the total number of pixels is less than num_pixels, one arbitrary pixel is repeated.
    The valid_pixels keeps track of true and repeated pixels.

    Args:
        num_pixels (int): Number of pixels to sample.
    """

    def __init__(self, num_pixels):
        self.num_pixels = num_pixels

    def __call__(self, sample):
        pixels = sample['pixels']
        T, C, S = pixels.shape
        if S > self.num_pixels:
            indices = random.sample(range(S), self.num_pixels)
            x = pixels[:, :, indices]
            valid_pixels = np.ones(self.num_pixels)
        elif S < self.num_pixels:
            x = np.zeros((T, C, self.num_pixels))
            x[..., :S] = pixels
            x[..., S:] = np.stack([x[:, :, 0] for _ in range(S, self.num_pixels)], axis=-1)
            valid_pixels = np.array([1 for _ in range(S)] + [0 for _ in range(S, self.num_pixels)])
        else:
            x = pixels
            valid_pixels = np.ones(self.num_pixels)
        # Repeat valid_pixels across time
        valid_pixels = np.repeat(valid_pixels[np.newaxis].astype(np.float32), x.shape[0], axis=0)
        sample['pixels'] = x
        sample['valid_pixels'] = valid_pixels
        return sample


class RandomSampleTimeSteps(object):
    """Randomly draw seq_length time steps to fix the time dimension.

    Args:
        seq_length (int): Number of time steps to sample. If -1, do nothing.
    """

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def __call__(self, sample):
        if self.seq_length == -1:
            return sample
        pixels, date_positions, valid_pixels = sample['pixels'], sample['positions'], sample['valid_pixels']
        t = pixels.shape[0]
        if t > self.seq_length:
            indices = sorted(random.sample(range(t), self.seq_length))
            sample['pixels'] = pixels[indices]
            sample['positions'] = date_positions[indices]
            sample['valid_pixels'] = valid_pixels[indices]
        else:
            raise NotImplementedError

        return sample

class RandomTemporalShift(object):
    """Randomly shift date positions

    Args:
        max_shift (int): Maximum possible temporal shift
    """

    def __init__(self, max_shift=60, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            sample['positions'] = sample['positions'] + shift


        return sample


class Normalize(object):
    """Normalize by rescaling pixels to [0, 1]

    Args:
        max_pixel_value (int): Max value of pixels to move pixels to [0, 1]
    """

    def __init__(self, max_pixel_value=65535):
        self.max_pixel_value = max_pixel_value

        # approximate max values
        max_parcel_box_m = 10000
        max_perimeter = max_parcel_box_m * 4
        max_area = max_parcel_box_m ** 2
        max_perimeter_area_ratio = max_perimeter
        max_cover_ratio = 1.0
        self.max_extra_values = np.array([max_perimeter, max_area, max_perimeter_area_ratio, max_cover_ratio])

    def __call__(self, sample):
        sample['pixels'] = np.clip(sample['pixels'], 0, self.max_pixel_value).astype(np.float32) / self.max_pixel_value
        if 'extra' in sample:
            sample['extra'] = sample['extra'].astype(np.float32) / self.max_extra_values
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample['pixels'] = torch.from_numpy(sample['pixels'].astype(np.float32))
        sample['valid_pixels'] = torch.from_numpy(sample['valid_pixels'].astype(np.float32))
        sample['positions'] = torch.from_numpy(sample['positions'].astype(np.long))
        if 'extra' in sample:
            sample['extra'] = torch.from_numpy(sample['extra'].astype(np.float32))
        if isinstance(sample['label'], int):
            sample['label'] = torch.tensor(sample['label']).long()
        return sample


