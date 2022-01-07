from tqdm import tqdm
import torch
import torch.nn as nn
import competitors.alda.loss as loss
from dataset import PixelSetData
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor
from utils.metrics import accuracy
from utils.train_utils import AverageMeter, cycle, to_cuda
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD
import numpy as np


def train_alda(
    model, config, writer, val_loader, device, best_model_path, fold_num, splits
):
    source_loader, target_loader = get_data_loaders(splits, config)

    if config.weights is not None:
        pretrained_path = f"{config.weights}/fold_{fold_num}"
        pretrained_weights = torch.load(f"{pretrained_path}/model.pt")["state_dict"]
        model.load_state_dict(pretrained_weights)

    features_dim = 128
    hidden_size = 1024
    class_num = config.num_classes
    ad_net = Multi_AdversarialNetwork(features_dim, hidden_size, class_num)
    ad_net.cuda()

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
            classifier_params + ad_net.get_parameters(),
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
            list(model.parameters()) + list(ad_net.parameters()),
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
            desc=f"ALDA Epoch {epoch + 1}/{config.epochs}",
        )

        losses = AverageMeter()
        class_accs = AverageMeter()
        domain_accs = AverageMeter()

        model.train()
        ad_net.train()

        for i in progress_bar:
            optimizer.zero_grad()
            x_s, x_t = next(source_iter), next(target_iter)
            labels_s = x_s["label"].cuda()

            y_s, f_s = model(*to_cuda(x_s, device), return_feats=True)
            y_t, f_t = model(*to_cuda(x_t, device), return_feats=True)
            features = torch.cat((f_s, f_t), dim=0)
            outputs = torch.cat((y_s, y_t), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            ad_out = ad_net(features)
            adv_loss, reg_loss, correct_loss = loss.ALDA_loss(ad_out, labels_s,
                    softmax_out, threshold=config.pseudo_threshold)

            adv_weight = config.trade_off
            trade_off = calc_coeff(i, high=1.0)

            transfer_loss = adv_weight * adv_loss + adv_weight * trade_off * correct_loss


            # reg_loss is only backward to the discriminator
            for param in model.parameters():
                param.requires_grad = False
            reg_loss.backward(retain_graph=True)
            for param in model.parameters():
                param.requires_grad = True

            cls_loss = criterion(y_s, labels_s)

            total_loss = cls_loss + transfer_loss

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.update(total_loss.item())
            class_accs.update(accuracy(y_s, labels_s), config.batch_size)

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

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class Multi_AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, class_num):
        super(Multi_AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        # self.max_iter = 10000.0
        self.max_iter = 1000.0

    def forward(self, x, grl=True):
        if self.training:
            self.iter_num += 1
        if grl and self.training:
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x = x * 1.0
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.0}]

  # def get_parameters(self):
  #   return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
