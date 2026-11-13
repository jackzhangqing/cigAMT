"""One fixed task objective shared by all trainable model families."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def task_loss(task,batch,out):
    if task=='noise_classification':
        return F.cross_entropy(out['logits'].float(),batch['class_id'].long())
    if task=='noise_localization':
        logits=out['logits'].float()
        truth=batch['noise_mask'].float().mean(dim=1,keepdim=True)
        if logits.shape[1]!=1:logits=logits.mean(dim=1,keepdim=True)
        bce=F.binary_cross_entropy_with_logits(logits,truth)
        p=logits.sigmoid();dims=(1,2)
        dice=1-((2*(p*truth).sum(dims)+1)/(p.sum(dims)+truth.sum(dims)+1)).mean()
        return bce+dice
    target=batch['target'].float()
    if task=='frequency_denoising':
        return (F.mse_loss(out['low'].float(),batch['target_low'].float())+
                F.mse_loss(out['high'].float(),batch['target_high'].float())+
                .5*F.mse_loss(out['prediction'].float(),target))
    error=(out['prediction'].float()-target).square()
    if task in ('forecasting','interpolation','imputation'):
        mask=batch['loss_mask'].float()
        # Average within each case first, so high missing ratios do not dominate.
        return ((error*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)).mean()
    return error.mean()
