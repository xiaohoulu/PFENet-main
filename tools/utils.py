import os
import random

import numpy as np
import torch
from thop import clever_format
from thop import profile

from .metrics import dice_score, iou_score, F2, mae_socre, acc_score, recall


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_poly(optimizer, init_lr, curr_iter, max_iter):
    warm_start_lr = 1e-7
    warm_steps = 1000

    if curr_iter <= warm_steps:
        warm_factor = (init_lr / warm_start_lr) ** (1 / warm_steps)
        warm_lr = warm_start_lr * warm_factor ** curr_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_lr
    else:
        lr = init_lr * (1 - (curr_iter - warm_steps) / (max_iter - warm_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def calculate_metrics(y_true, y_pred):
    b, c, h, w = y_true.size()
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred.astype(np.float64)

    y_true[y_true > 0.5] = 1
    y_true[y_true != 1] = 0

    y_p = y_pred.reshape(h, w)
    y_t = y_true.reshape(h, w)

    # score_smeasure = sm_score(y_t, y_p)
    # score_fmeasure = wfm_score(y_t, y_p)
    score_mae = mae_socre(y_t, y_p)
    # score_emeasure = em_score(y_t, y_p)

    # y_pred = y_pred.astype(np.uint8)
    # y_true = y_true.astype(np.uint8)
    #
    # y_pred = y_pred.reshape(-1)
    # y_true = y_true.reshape(-1)

    ## Score
    score_f1 = dice_score(y_true, y_pred)
    score_iou = iou_score(y_true, y_pred)
    score_f2 = F2(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_acc = acc_score(y_true, y_pred)

    return [score_f1, score_iou, score_f2, score_recall, score_acc, score_mae]
