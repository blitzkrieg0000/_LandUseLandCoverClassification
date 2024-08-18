import torch
import torch.nn.functional as F
import torch.nn as nn


def CustomCrossEntropyloss(pred, target):
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    summ = -(target * log_prob).sum(dim=1)
    return summ.mean()


def DiceLoss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()


def CombinedLoss(pred, target):
    """Dice + CrossEntropyLoss"""
    return DiceLoss(pred, target) + CustomCrossEntropyloss(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: BATCHxCLASSx128x128
        # targets: BATCHx128x128 (her piksel için sınıf etiketleri)

        inputs = F.softmax(inputs, dim=1)  # BATCHxCLASSx128x128
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])  # BATCHx128x128xCLASS
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # BATCHxCLASSx128x128

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss