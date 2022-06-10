import torch.nn.functional as F
import torch

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=20):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        
def adjust_learning_rate(optimizer, gam, epoch, step_index, iteration, epoch_size):
    warmup_epoch = 20
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gam ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

## metrics from other papers



##
def kl_loss(pred, mask):

    pred = pred.view(pred.size()[0], -1)
    pred = torch.sigmoid(pred)
    mask = mask.view(mask.size()[0], -1)

    log_pred = F.log_softmax(pred, dim=1)
    # log_pred = torch.log(log_pred / torch.norm(log_pred, p=1, dim=1, keepdim=True))
    p_mask = F.softmax(mask, dim=1)
    kl_loss = 100*F.kl_div(log_pred, p_mask, reduction='batchmean')

    return kl_loss

def cross_entropy_with_weight(logits, labels):
    logits = logits.sigmoid().view(-1)
    labels = labels.view(-1)
    eps = 1e-6 # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]

    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()

    return cross_entropy

def criterion(pred, labels):
    celoss = cross_entropy_with_weight(pred,labels)
    klloss = kl_loss(pred,labels)
    loss  = 0.01*celoss + klloss

    return loss
