from typing import List
import numpy as np
import torch


class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0, checkpoint_save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.Inf
        self.checkpoint_save_path = checkpoint_save_path


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping patience counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_save_path)
        self.val_loss_min = val_loss



# =================================================================================================================== #
#! Functions
# =================================================================================================================== #
def ChangeMaskOrder(mask: torch.Tensor, classes: torch.Tensor):
    extra_classes = classes.unique()
    others = extra_classes[~torch.isin(extra_classes, classes)]
    other_maps = {x:0 for x in others}
    mapping = {x:i for i, x in enumerate(classes)}
    mapping.update(other_maps)

    new_mask = mask.clone()
    for old, new in mapping.items():
        new_mask[mask == old] = new
    return new_mask


def CountModelParameters(model):
    """ Count Model Parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)