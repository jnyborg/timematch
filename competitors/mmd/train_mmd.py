'''
Re-implementation of MMD loss (DAN) from https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/classification/dan.py
'''
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD
from torchvision import transforms
from tqdm import tqdm

from competitors.mmd.dan import MultipleKernelMaximumMeanDiscrepancy
from competitors.mmd.kernels import GaussianKernel
from dataset import PixelSetData
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor, RandomTemporalShift, Identity
from utils.train_utils import AverageMeter, cat_samples, cycle, to_cuda
from utils.metrics import accuracy


def train_mmd(model, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader = get_data_loaders(splits, config)

    model.to(device)
    if config.weights is not None:
        pretrained_path = f"{config.weights}/fold_{fold_num}"
        pretrained_weights = torch.load(f"{pretrained_path}/model.pt")["state_dict"]
        model.load_state_dict(pretrained_weights)

    linear_time = True  # whether to compute MMD in linear time or quadratic
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=linear_time
    )
    if config.use_default_optim:
        base_lr = 1.0
        classifier_params = [
            {"params": model.spatial_encoder.parameters(), "lr": 0.1 * base_lr},
            {"params": model.temporal_encoder.parameters(), "lr": 0.1 * base_lr},
            {"params": model.decoder.parameters(), "lr": 1.0 * base_lr},
        ]
        # lr_gamma = 0.0003
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
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * config.steps_per_epoch, eta_min=0
        )

    ## train
    best_f1 = 0.0
    criterion = nn.CrossEntropyLoss()
    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    global_step = 0
    for epoch in range(config.epochs):
        progress_bar = tqdm(range(config.steps_per_epoch), desc=f'MMD Epoch {epoch + 1}/{config.epochs}')

        losses = AverageMeter()
        class_accs = AverageMeter()

        model.train()
        mkmmd_loss.train()

        for _ in progress_bar:
            x_s, x_t = next(source_iter), next(target_iter)
            labels_s = x_s["label"].cuda()

            x = cat_samples([x_s, x_t])
            y, f = model(*to_cuda(x, device), return_feats=True)
            y_s, _ = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            cls_loss = criterion(y_s, labels_s)
            transfer_loss = mkmmd_loss(f_s, f_t)
            loss = cls_loss + transfer_loss * config.trade_off

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.update(loss.item(), config.batch_size)
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
        best_f1 = validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer)

    # save final model and use for evaluation
    torch.save({'state_dict': model.state_dict()}, best_model_path)


def get_data_loaders(splits, config):
    def create_data_loader(dataset):
        return torch.utils.data.DataLoader(dataset,
                num_workers=config.num_workers, pin_memory=True,
                batch_size=config.batch_size, shuffle=True, drop_last=True)

    train_transform = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source, config.classes, train_transform, indices=splits[config.source]['train'])
    source_loader = create_data_loader(source_dataset)
    target_dataset = PixelSetData(config.data_root, config.target, config.classes, train_transform, indices=splits[config.target]['train'])
    target_loader = create_data_loader(target_dataset)

    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    print(f'size of target dataset: {len(target_dataset)} ({len(target_loader)} batches)')

    return source_loader, target_loader
