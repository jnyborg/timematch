"""
Re-implementation of DANN from https://github.com/thuml/Transfer-Learning-Library
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor, RandomTemporalShift, Identity
from utils.metrics import accuracy
from utils.train_utils import AverageMeter, cycle, to_cuda, cat_samples
from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
from competitors.dann.grl import WarmStartGradientReverseLayer
import numpy as np


def train_dann(
    model, config, writer, val_loader, device, best_model_path, fold_num, splits
):
    source_loader, target_loader = get_data_loaders(splits, config)

    if config.weights is not None:
        pretrained_path = f"{config.weights}/fold_{fold_num}"
        pretrained_weights = torch.load(f"{pretrained_path}/model.pt")["state_dict"]
        model.load_state_dict(pretrained_weights)

    features_dim = 128
    hidden_size = 1024
    if config.adv_loss == "DANN":
        domain_discri = DomainDiscriminator(features_dim, hidden_size=hidden_size).to(
            device
        )
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    elif config.adv_loss in ["CDAN", "CDAN+E"]:
        use_entropy = config.adv_loss == "CDAN+E"
        use_randomized = features_dim * config.num_classes > 4096
        randomized_dim = 1024
        if use_randomized:
            domain_discri = DomainDiscriminator(
                randomized_dim, hidden_size=hidden_size
            ).to(device)
        else:
            domain_discri = DomainDiscriminator(
                features_dim * config.num_classes, hidden_size=hidden_size
            ).to(device)

        domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri,
            entropy_conditioning=use_entropy,
            num_classes=config.num_classes,
            features_dim=features_dim,
            randomized=use_randomized,
            randomized_dim=randomized_dim,
        ).to(device)
    else:
        raise NotImplementedError

    if config.use_default_optim:
        base_lr = 1.0
        classifier_params = [
            {"params": model.spatial_encoder.parameters(), "lr": 0.1 * base_lr},
            {"params": model.temporal_encoder.parameters(), "lr": 0.1 * base_lr},
            {"params": model.decoder.parameters(), "lr": 1.0 * base_lr},
        ]
        lr_gamma = 0.001
        lr_decay = 0.75
        optimizer = SGD(
            classifier_params + domain_discri.get_parameters(),
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
            list(model.parameters()) + list(domain_discri.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * config.steps_per_epoch, eta_min=0
        )

    ## train
    best_f1 = 0.0
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))

    for epoch in range(config.epochs):
        progress_bar = tqdm(
            range(config.steps_per_epoch),
            desc=f"{config.adv_loss} Epoch {epoch + 1}/{config.epochs}",
        )

        losses = AverageMeter()
        class_accs = AverageMeter()
        domain_accs = AverageMeter()

        model.train()
        domain_adv.train()

        for _ in progress_bar:
            x_s, x_t = next(source_iter), next(target_iter)
            labels_s = x_s["label"].cuda()

            x = cat_samples([x_s, x_t])
            y, f = model(*to_cuda(x, device), return_feats=True)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            cls_loss = criterion(y_s, labels_s)
            if config.adv_loss == "DANN":
                transfer_loss = domain_adv(f_s, f_t)
            else:
                transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * config.trade_off

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.update(loss.item(), config.batch_size)
            class_accs.update(accuracy(y_s, labels_s), config.batch_size)
            domain_accs.update(domain_acc, config.batch_size)

            progress_bar.set_postfix(
                loss=f"{losses.avg:.3f}",
                class_acc=f"{class_accs.avg:.2f}",
                domain_acc=f"{domain_accs.avg:.2f}",
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
        best_f1 = validation(
            best_f1, None, config, criterion, device, epoch, model, val_loader, writer
        )

    # save final model and use for evaluation
    torch.save({"state_dict": model.state_dict()}, best_model_path)


def get_data_loaders(splits, config):
    def create_data_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    train_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )

    source_dataset = PixelSetData(
        config.data_root,
        config.source,
        config.classes,
        train_transform,
        indices=splits[config.source]["train"],
    )
    source_loader = create_data_loader(source_dataset)
    target_dataset = PixelSetData(
        config.data_root,
        config.target,
        config.classes,
        train_transform,
        indices=splits[config.target]["train"],
    )
    target_loader = create_data_loader(target_dataset)

    print(
        f"size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)"
    )
    print(
        f"size of target dataset: {len(target_dataset)} ({len(target_loader)} batches)"
    )

    return source_loader, target_loader


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.0}]


class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(f_j^t)].
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(
        self,
        domain_discriminator: nn.Module,
        reduction: Optional[str] = "mean",
        grl: Optional = None,
    ):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = (
            WarmStartGradientReverseLayer(
                alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True
            )
            if grl is None
            else grl
        )
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: F.binary_cross_entropy(
            input, target, weight=weight, reduction=reduction
        )
        self.domain_discriminator_accuracy = None

    def forward(
        self,
        f_s: torch.Tensor,
        f_t: torch.Tensor,
        w_s: Optional[torch.Tensor] = None,
        w_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (
            binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t)
        )

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (
            self.bce(d_s, d_label_s, w_s.view_as(d_s))
            + self.bce(d_t, d_label_t, w_t.view_as(d_t))
        )


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_
    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(T(f_j^t, g_j^t))],\\
    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.
    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(
        self,
        domain_discriminator: nn.Module,
        entropy_conditioning: Optional[bool] = False,
        randomized: Optional[bool] = False,
        num_classes: Optional[int] = -1,
        features_dim: Optional[int] = -1,
        randomized_dim: Optional[int] = 1024,
        reduction: Optional[str] = "mean",
    ):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True
        )
        self.entropy_conditioning = entropy_conditioning

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(
                features_dim, num_classes, randomized_dim
            )
        else:
            self.map = MultiLinearMap()

        self.bce = (
            lambda input, target, weight: F.binary_cross_entropy(
                input, target, weight, reduction=reduction
            )
            if self.entropy_conditioning
            else F.binary_cross_entropy(input, target, reduction=reduction)
        )
        self.domain_discriminator_accuracy = None

    def forward(
        self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor
    ) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        d_label = torch.cat(
            (
                torch.ones((g_s.size(0), 1)).to(g_s.device),
                torch.zeros((g_t.size(0), 1)).to(g_t.device),
            )
        )
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        return self.bce(d, d_label, weight.view_as(d))


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map
    Given two inputs :math:`f` and :math:`g`, the definition is
    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),
    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.
    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024
    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(
        self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024
    ):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map
    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


def entropy(predictions: torch.Tensor, reduction="none") -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:
    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c
    where C is number of classes.
    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``
    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == "mean":
        return H.mean()
    else:
        return H
