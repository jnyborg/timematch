import torch
import numpy as np
from tabulate import tabulate


@torch.no_grad()
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return preds.eq(targets).float().mean().item()


def f1_score(confusion_matrix, reduce_mean=True):
    f1_scores = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * float(true_positives) / denom
        f1_scores.append(f1_score)
    return np.mean(f1_scores) if reduce_mean else f1_scores


def precision_recall_fscore_support(confusion_matrix):
    f1_scores, precisions, recalls, supports = [], [], [], []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives

        denom = true_positives + false_positives
        precision = 0.0 if denom == 0 else float(true_positives) / denom

        denom = true_positives + false_negatives
        recall = 0.0 if denom == 0 else float(true_positives) / denom

        denom = precision + recall
        f1_score = 0.0 if denom == 0 else 2 * float(precision * recall) / denom

        f1_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)
        supports.append(int(true_positives + false_negatives))

    return precisions, recalls, f1_scores, supports


def accuracy_cm(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def compute_confusion_matrix(prediction, ground_truth, num_classes):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes), (0, num_classes)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def classification_report(cm, class_names):
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(cm)
    rows = list(zip(class_names, precisions, recalls, f1_scores, supports))
    accuracy = accuracy_cm(cm)
    total_support = sum(supports)
    rows.append([None, None, None, None, None])
    rows.append(['accuracy', None, None, accuracy, total_support])
    rows.append(['macro avg', np.mean(precisions), np.mean(recalls), np.mean(f1_scores), total_support])
    # weights = np.array(supports) / total_support
    # rows.append(['weighted avg', np.average(precisions, weights=weights), np.average(recalls, weights=weights), np.average(f1_scores, weights=weights), total_support])
    headers = ['', 'precision', 'recall', 'f1-score', 'support']
    return tabulate(rows, headers=headers, floatfmt='.2f', tablefmt='pipe')


def confusion_matrix_report(cm, class_names):
    rows = []
    classes = list(range(len(class_names)))
    for cls, conf in zip(classes, cm):
        rows.append([cls, ] + conf.tolist())
    headers = ['true\\pred', ] + classes
    return tabulate(rows, headers=headers, floatfmt='.2f', tablefmt='pipe')


def overall_classification_report(cms, class_names):
    class_metrics = [precision_recall_fscore_support(cm) for cm in cms]
    class_metrics = np.array(class_metrics)  # (len(cms), 4, len(class_names))
    class_metrics[:, :-1] *= 100.0
    rows = []
    for class_idx, class_name in enumerate(class_names):
        metrics = class_metrics[:, :, class_idx]
        mean_stds = list(zip(np.mean(metrics, axis=0), np.std(metrics, axis=0)))
        support_mean, support_std = mean_stds[-1]
        # mean_stds = np.array(mean_stds[:-1]) * 100.0
        mean_stds = [f'{mean:.1f}±{std:.1f}' for mean, std in mean_stds[:-1]] + [f'{support_mean:.1f}±{support_std:.1f}']
        rows.append((class_name, *mean_stds))

    accs = np.array([accuracy_cm(cm) * 100.0 for cm in cms])
    accuracy = '{:.1f}±{:.1f}'.format(np.mean(accs), np.std(accs))
    rows.append([None, None, None, None, None])
    rows.append(['accuracy', None, None, None, accuracy])
    macro_avg = []
    for i in range(4):
        if i == 3:  # sum support
            macro_avg_per_run = np.sum(class_metrics[:, i, :], axis=-1)
            std = np.std(macro_avg_per_run)
        else:
            macro_avg_per_run = np.mean(class_metrics[:, i, :], axis=-1)
            std = np.std(macro_avg_per_run)
        macro_avg.append(f'{np.mean(macro_avg_per_run):.1f}±{std:.1f}')
    rows.append(['macro avg', *macro_avg])
    headers = ['', 'precision', 'recall', 'f1-score', 'support']
    return tabulate(rows, headers=headers, floatfmt='.2f', tablefmt='pipe')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
