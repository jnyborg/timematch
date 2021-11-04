import os
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics
from utils.train_utils import AverageMeter, to_cuda


def validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer, temporal_shift=None):
    val_metrics = evaluation(model, val_loader, device, config.classes, criterion, mode='val', temporal_shift=temporal_shift)
    val_loss, val_acc, val_f1, val_kappa = val_metrics['loss'], val_metrics['accuracy'], val_metrics['macro_f1'], val_metrics['kappa']
    writer.add_scalar('val/loss', val_loss, global_step=epoch)
    writer.add_scalar('val/accuracy', val_acc, global_step=epoch)
    writer.add_scalar('val/f1', val_f1, global_step=epoch)
    writer.add_scalar('val/kappa', val_kappa, global_step=epoch)
    print(f"Validation result: loss={val_loss:.4f}, acc={val_acc:.2f}, f1={val_f1:.4f}")
    if val_f1 > best_f1:
        print(f'Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}!')
        best_f1 = val_f1
        if best_model_path is not None:
            print(f'Saving best model to {best_model_path}')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_f1': best_f1}, best_model_path)
    else:
        print(f'Validation F1 did not improve from {best_f1:.4f}.')
    return best_f1


@torch.no_grad()
def evaluation(model, data_loader, device, class_names, criterion=None, mode='val', temporal_shift=None):
    y_true, y_pred = [], []

    loss_meter = AverageMeter()

    model.eval()
    for sample in tqdm(data_loader, desc='Validating' if mode == 'val' else 'Testing'):
        target = sample['label']
        y_true.extend(target.tolist())
        target = target.cuda(device=device, non_blocking=True)

        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        if temporal_shift is not None:
            logits = model.forward(pixels, valid_pixels, positions + temporal_shift, extra)
        else:
            logits = model.forward(pixels, valid_pixels, positions, extra)

        predictions = logits.argmax(dim=1)

        if criterion is not None:
            loss = criterion(logits, target)
            loss_meter.update(loss.item(), n=pixels.size(0))
        y_pred.extend(predictions.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'loss': loss_meter.avg,
        'macro_f1': sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': sklearn.metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': sklearn.metrics.cohen_kappa_score(y_true, y_pred, labels=list(range(len(class_names)))),
        'classification_report': sklearn.metrics.classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0),
        'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
   }

    return metrics
