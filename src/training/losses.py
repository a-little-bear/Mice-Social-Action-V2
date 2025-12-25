import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MacroSoftF1Loss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super(MacroSoftF1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=0)
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)
        
        f1 = (2 * tp) / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1.mean()

class OHEMLoss(nn.Module):
    def __init__(self, rate=0.7, base_loss=None):
        super(OHEMLoss, self).__init__()
        self.rate = rate
        self.base_loss = base_loss if base_loss is not None else nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        loss = self.base_loss(logits, targets)
        
        num_examples = loss.numel()
        num_hard = int(self.rate * num_examples)
        
        if num_hard == 0:
            return loss.mean()
            
        loss_flat = loss.view(-1)
        loss_sorted, _ = torch.topk(loss_flat, num_hard)
        return loss_sorted.mean()
