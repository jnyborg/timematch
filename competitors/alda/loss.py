import numpy as np
import torch
import torch.nn as nn


def create_matrix(n):
    """
    :param n: matrix size (class num)
    :return a matrix with torch.tensor type:
    for example n=3:
    1     -1/2  -1/2
    -1/2    1   -1/2
    -1/2  -1/2    1
    """
    a = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j:
                a[i,j]=1
            else:
                a[i,j]=-1/(n-1)
    return torch.from_numpy(a).cuda()

def ALDA_loss(ad_out_score, labels_source, softmax_out, threshold=0.9):
    """
    :param ad_out_score: the discriminator output (N, C, H, W)
    :param labels_source: the source ground truth (N, H, W)
    :param softmax_out: the model prediction probability (N, C, H, W)
    :return: 
    adv_loss: adversarial learning loss
    reg_loss: regularization term for the discriminator
    correct_loss: corrected self-training loss
    """
    ad_out = torch.sigmoid(ad_out_score)

    batch_size = ad_out.size(0) // 2
    class_num = ad_out.size(1)

    labels_source_mask = torch.zeros(batch_size, class_num).to(ad_out.device).scatter_(1, labels_source.unsqueeze(1), 1)
    probs_source = softmax_out[:batch_size].detach()
    probs_target = softmax_out[batch_size:].detach()
    maxpred, argpred = torch.max(probs_source, dim=1)
    preds_source_mask = torch.zeros(batch_size, class_num).to(ad_out.device).scatter_(1, argpred.unsqueeze(1), 1)
    maxpred, argpred = torch.max(probs_target, dim=1)
    preds_target_mask = torch.zeros(batch_size, class_num).to(ad_out.device).scatter_(1, argpred.unsqueeze(1), 1)

    # filter out those low confidence samples
    target_mask = (maxpred > threshold)
    preds_target_mask = torch.where(target_mask.unsqueeze(1), preds_target_mask, torch.zeros(1).to(ad_out.device))
    # construct the confusion matrix from ad_out. See the paper for more details.
    confusion_matrix = create_matrix(class_num)
    ant_eye = (1-torch.eye(class_num)).cuda().unsqueeze(0)
    confusion_matrix = ant_eye/(class_num-1) + torch.mul(confusion_matrix.unsqueeze(0), ad_out.unsqueeze(1)) #(2*batch_size, class_num, class_num)
    preds_mask = torch.cat([preds_source_mask, preds_target_mask], dim=0) #labels_source_mask
    loss_pred = torch.mul(confusion_matrix, preds_mask.unsqueeze(1)).sum(dim=2)
    # different correction targets for different domains
    loss_target = (1 - preds_target_mask) / (class_num-1)
    loss_target = torch.cat([labels_source_mask, loss_target], dim=0)
    if not ((loss_pred>=0).all() and (loss_pred<=1).all()):
        raise AssertionError
    mask = torch.cat([(maxpred >= 0), target_mask], dim=0)
    adv_loss = nn.BCELoss(reduction='none')(loss_pred, loss_target)[mask]
    adv_loss = torch.sum(adv_loss) / mask.float().sum()
    
    # reg_loss
    reg_loss = nn.CrossEntropyLoss()(ad_out_score[:batch_size], labels_source)
    
    # corrected target loss function
    target_probs = 1.0*softmax_out[batch_size:]
    correct_target = torch.mul(confusion_matrix.detach()[batch_size:], preds_target_mask.unsqueeze(1)).sum(dim=2)
    correct_loss = -torch.mul(target_probs, correct_target)
    correct_loss = torch.mean(correct_loss[target_mask])
    return adv_loss, reg_loss, correct_loss
