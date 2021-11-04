from collections import defaultdict
import datetime as dt
import os
import pickle as pkl
from typing import List

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import transforms
import zarr

from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
)
from utils import label_utils


class PixelSetData(data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_name,
        classes,
        transform=None,
        indices=None,
        with_extra=False,
    ):
        super(PixelSetData, self).__init__()

        self.folder = os.path.join(data_root, dataset_name)
        self.dataset_name = dataset_name  # country/tile/year
        self.country = dataset_name.split("/")[-3]
        self.tile = dataset_name.split("/")[-2]
        self.data_folder = os.path.join(self.folder, "data")
        self.meta_folder = os.path.join(self.folder, "meta")
        self.transform = transform
        self.with_extra = with_extra

        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.samples, self.metadata = self.make_dataset(
            self.data_folder, self.meta_folder, self.class_to_idx, indices, self.country
        )

        self.dates = self.metadata["dates"]
        self.date_positions = self.days_after(self.metadata["start_date"], self.dates)
        self.date_indices = np.arange(len(self.date_positions))

    def get_shapes(self):
        return [
            (len(self.dates), 10, parcel["n_pixels"])
            for parcel in self.metadata["parcels"]
        ]

    def get_labels(self):
        return np.array([x[2] for x in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, parcel_idx, y, extra = self.samples[index]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": index,
            "parcel_index": parcel_idx,  # mapping to metadata
            "pixels": pixels,
            "valid_pixels": np.ones(
                (pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "positions": np.array(self.date_positions),
            "extra": np.array(extra),
            "label": y,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_dataset(self, data_folder, meta_folder, class_to_idx, indices, country):
        metadata = pkl.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))

        instances = []
        new_parcel_metadata = []

        code_to_class_name = label_utils.get_code_to_class(country)

        unknown_crop_codes = set()

        for parcel_idx, parcel in enumerate(metadata["parcels"]):
            if indices is not None:
                if not parcel_idx in indices:
                    continue
            crop_code = parcel["label"]
            if country == "austria":
                crop_code = int(crop_code)
            parcel_path = os.path.join(data_folder, f"{parcel_idx}.zarr")
            if crop_code not in code_to_class_name:
                unknown_crop_codes.add(crop_code)
            class_name = code_to_class_name.get(crop_code, "unknown")
            class_index = class_to_idx.get(class_name, class_to_idx["unknown"])
            extra = parcel['geometric_features']

            item = (parcel_path, parcel_idx, class_index, extra)
            instances.append(item)
            new_parcel_metadata.append(parcel)

        for crop_code in unknown_crop_codes:
            print(
                f"Parcels with crop code {crop_code} was not found in .yml class mapping and was assigned to unknown."
            )

        metadata["parcels"] = new_parcel_metadata

        assert len(metadata["parcels"]) == len(instances)

        return instances, metadata

    def days_after(self, start_date, dates):
        def parse(date):
            d = str(date)
            return int(d[:4]), int(d[4:6]), int(d[6:])

        def interval_days(date1, date2):
            return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)

        date_positions = [interval_days(d, start_date) for d in dates]
        return date_positions

    def get_unknown_labels(self):
        """
        Reports the categorization of crop codes for this dataset
        """
        class_count = defaultdict(int)
        class_parcel_size = defaultdict(float)
        # metadata = pkl.load(open(os.path.join(self.meta_folder, 'metadata.pkl'), 'rb'))
        metadata = self.metadata
        for meta in metadata["parcels"]:
            class_count[meta["label"]] += 1
            class_parcel_size[meta["label"]] += meta["n_pixels"]

        class_avg_parcel_size = {
            cls: total_px / class_count[cls]
            for cls, total_px in class_parcel_size.items()
        }

        code_to_class_name = label_utils.get_code_to_class(self.country)
        codification_table = label_utils.get_codification_table(self.country)
        unknown = []
        known = defaultdict(list)
        for code, count in class_count.items():
            avg_pixels = class_avg_parcel_size[code]
            if self.country == "denmark":
                code = int(code)
            code_name = codification_table[str(code)]
            if code in code_to_class_name:
                known[code_to_class_name[code]].append(
                    (code, code_name, count, avg_pixels)
                )
            else:
                unknown.append((code, code_name, count, avg_pixels))

        print("\nCategorized crop codes:")
        for class_name, codes in known.items():
            total_parcels = sum(x[2] for x in codes)
            avg_parcel_size = sum(x[3] for x in codes) / len(codes)
            print(f"{class_name} (n={total_parcels}, avg size={avg_parcel_size:.3f}):")
            codes = reversed(sorted(codes, key=lambda x: x[2]))
            for code, code_name, count, avg_pixels in codes:
                print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")
        unknown = reversed(sorted(unknown, key=lambda x: x[2]))
        print("\nUncategorized crop codes:")
        for code, code_name, count, avg_pixels in unknown:
            print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_train_loader(ds, batch_size, num_workers):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def create_evaluation_loaders(dataset_name, splits, config, sample_pixels_val=False):
    """
    Create data loaders for unsupervised domain adaptation
    """

    is_tsnet = config.model == "tsnet"
    # Validation dataset
    val_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels) if sample_pixels_val else Identity(),
            RandomSampleTimeSteps(config.seq_length) if is_tsnet else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    val_dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        val_transform,
        indices=splits[dataset_name]["val"],
    )
    val_loader = data.DataLoader(
        val_dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(
            val_dataset, config.batch_size, by_pixel_dim=not sample_pixels_val
        ),
    )

    # Test dataset
    test_transform = transforms.Compose(
        [
            RandomSampleTimeSteps(config.seq_length) if is_tsnet else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    test_dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        test_transform,
        indices=splits[dataset_name]["test"],
    )
    test_loader = data.DataLoader(
        test_dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(test_dataset, config.batch_size),
    )

    print(f"evaluation dataset:", dataset_name)
    print(f"val target data: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"test taget data: {len(test_dataset)} ({len(test_loader)} batches)")

    return val_loader, test_loader


class GroupByShapesBatchSampler(torch.utils.data.BatchSampler):
    """
    Group parcels by their time and/or pixel dimension, allowing for batches
    with varying dimensionality.
    """

    def __init__(self, data_source, batch_size, by_time=True, by_pixel_dim=True):
        self.batches = []
        self.data_source = data_source

        datasets: List[PixelSetData] = []
        # shapes[index] contains data_source[index] (seq_length, n_channels, n_pixels)
        if isinstance(data_source, PixelSetData):
            datasets = [data_source]
            shapes = data_source.get_shapes()
        elif isinstance(data_source, ConcatDataset):
            datasets = data_source.datasets
            shapes = [shape for d in datasets for shape in d.get_shapes()]
        elif isinstance(data_source, Subset):
            datasets = [data_source]
            if isinstance(data_source.dataset, ConcatDataset):
                shapes = [
                    shape
                    for d in data_source.dataset.datasets
                    for shape in d.get_shapes()
                ]
                shapes = [
                    shape
                    for idx, shape in enumerate(shapes)
                    if idx in data_source.indices
                ]
            else:
                shapes = [
                    shape
                    for idx, shape in enumerate(data_source.dataset.get_shapes())
                    if idx in data_source.indices
                ]
        else:
            raise NotImplementedError

        # group indices by (seq_length, n_pixels)
        shp_to_indices = defaultdict(list)  # unique shape -> sample indices
        for idx, shp in enumerate(shapes):
            key = []
            if by_time:
                key.append(shp[0])
            if by_pixel_dim:
                key.append(shp[2])
            shp_to_indices[tuple(key)].append(idx)

        # create batches grouped by shape
        batches = []
        for indices in shp_to_indices.values():
            if len(indices) > batch_size:
                batches.extend(
                    [
                        indices[i : i + batch_size]
                        for i in range(0, len(indices), batch_size)
                    ]
                )
            else:
                batches.append(indices)

        self.batches = batches
        self.datasets = datasets
        self.batch_size = batch_size
        # self._unit_test()

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def _unit_test(self):
        # make sure that we iterate across all items
        # 1) no duplicates
        assert sum(len(batch) for batch in self.batches) == sum(
            len(d) for d in self.datasets
        )
        # 2) all indices are present
        assert set([idx for indices in self.batches for idx in indices]) == set(
            range(sum(len(d) for d in self.datasets))
        )

        # make sure that no batch is larger than batch size
        assert all(len(batch) <= self.batch_size for batch in self.batches)


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels))
        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print(f"using batch size={batch_size} for balanced batch sampler")

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]


if __name__ == "__main__":
    classes = label_utils.get_classes("france")
    dataset = PixelSetData("/media/data/mark_pixels", "france/31TCJ/2017", classes, with_extra=True)
    print(dataset[0])
    print(dataset.date_positions)
