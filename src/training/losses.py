import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        focal_term = (1 - pt) ** self.gamma
        
        if self.pos_weight is not None:
            if self.pos_weight.device != inputs.device:
                self.pos_weight = self.pos_weight.to(inputs.device)
                
            weight = torch.where(targets == 1, self.pos_weight, torch.ones_like(self.pos_weight))
            loss = weight * focal_term * bce_loss
        else:
            loss = self.alpha * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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

    def forward(self, logits, targets, mask=None):
        loss = self.base_loss(logits, targets)
        
        if mask is not None:
            if mask.shape != loss.shape:
                if mask.dim() < loss.dim():
                    while mask.dim() < loss.dim():
                        mask = mask.unsqueeze(-1)
                    mask = mask.expand_as(loss)
            
            loss_flat = loss[mask.bool()]
        else:
            loss_flat = loss.view(-1)
        
        num_examples = loss_flat.numel()
        if num_examples == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        num_hard = int(self.rate * num_examples)
        
        if num_hard == 0:
            return loss_flat.mean()
            
        loss_sorted, _ = torch.topk(loss_flat, num_hard)
        return loss_sorted.mean()
