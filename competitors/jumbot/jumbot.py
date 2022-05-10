"""
Dependances : 
- python (3.8.0)
- numpy (1.19.2)
- torch (1.7.1)
- POT (0.7.0)
- Cuda

command:
python3 train.py
"""


import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD
from torch.utils import data
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor, RandomTemporalShift, Identity
from utils.metrics import accuracy
from utils.train_utils import AverageMeter, cat_samples, cycle, to_cuda


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_jumbot(model, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader = get_data_loaders(splits, config, source_balanced=False)

    model.to(device)
    if config.weights is not None:
        pretrained_path = f"{config.weights}/fold_{fold_num}"
        pretrained_weights = torch.load(f"{pretrained_path}/model.pt")["state_dict"]
        model.load_state_dict(pretrained_weights)
        print('using pretrained weights', config.weights)


    base_lr = 1.0
    classifier_params = [
        {"params": model.spatial_encoder.parameters(), "lr": 0.1 * base_lr},
        {"params": model.temporal_encoder.parameters(), "lr": 0.1 * base_lr},
        {"params": model.decoder.parameters(), "lr": 1.0 * base_lr},
    ]
    lr_gamma = 0.001
    lr_decay = 0.75
    optimizer = SGD(
        classifier_params,
        config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    lr_scheduler = LambdaLR(
        optimizer, lambda x: config.lr * (1.0 + lr_gamma * float(x)) ** (-lr_decay)
    )

    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_f1 = 0.0

    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    for epoch in range(config.epochs):
        progress_bar = tqdm(range(config.steps_per_epoch), desc=f'JUMBOT Epoch {epoch + 1}/{config.epochs}')

        losses = AverageMeter()
        class_accs = AverageMeter()

        model.train()
        for _ in progress_bar:
            x_s, x_t = next(source_iter), next(target_iter)
            labels_s = x_s["label"].cuda()

            y_s, f_s = model(*to_cuda(x_s, device), return_feats=True)
            y_t, f_t = model(*to_cuda(x_t, device), return_feats=True)
            pred_x_t = F.softmax(y_t, 1)

            cls_loss = criterion(y_s, labels_s)

            one_hot_labels_s = F.one_hot(labels_s, num_classes=config.num_classes).float()
            M_embed = torch.cdist(f_s, f_t)**2  # term on embedded data
            M_sce = - torch.mm(one_hot_labels_s, torch.transpose(torch.log(pred_x_t), 0, 1))  # term on labels
            M = config.eta1 * M_embed + config.eta2 * M_sce  
            # M_normalized = M / M.max()  # normalize by max to avoid numerical issues

            #OT computation
            a, b = ot.unif(f_s.size()[0]), ot.unif(f_t.size()[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), config.epsilon, config.tau)
            # To get DeepJDOT (https://arxiv.org/abs/1803.10081) comment the line above 
            # and uncomment the following line:
            #pi = ot.emd(a, b, M.detach().cpu().numpy())
            pi = torch.from_numpy(pi).float().cuda() # Transport plan between minibatches
            transfer_loss = torch.sum(pi * M)

            # if global_step % 100 == 0:
            #     print(torch.sum(pi), transfer_loss, torch.min(M), torch.min(M_embed), torch.min(M_sce))

            # train the model 
            tot_loss = cls_loss + transfer_loss
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.update(tot_loss.item(), config.batch_size)
            class_accs.update(accuracy(y_s, labels_s), config.batch_size)

            progress_bar.set_postfix(
                loss=f"{losses.avg:.3f}",
                class_acc=f"{class_accs.avg:.2f}",
            )
            if global_step % config.log_step == 0:
                writer.add_scalar("train/loss", losses.val, global_step)
                writer.add_scalar("train/accuracy", class_accs.val, global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
            global_step += 1

        progress_bar.close()
        model.eval()
        best_f1 = validation(best_f1, None, config, criterion, device, epoch, model, val_loader, writer)

    # save final model and use for evaluation
    torch.save({'state_dict': model.state_dict()}, best_model_path)


def get_data_loaders(splits, config, source_balanced=False):
    train_transform = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source, config.classes, train_transform, indices=splits[config.source]['train'])

    if source_balanced:
        print("using balanced loader for source")
        source_labels = source_dataset.get_labels()
        train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=config.batch_size)

        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_sampler=train_batch_sampler,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )


    target_dataset = PixelSetData(config.data_root, config.target, config.classes, train_transform, indices=splits[config.target]['train'])
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    print(f'size of target dataset: {len(target_dataset)} ({len(target_loader)} batches)')

    return source_loader, target_loader


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels))
        print(classes)

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
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

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
