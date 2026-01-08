import torch
import torch.nn as nn
from copy import deepcopy

class EMA:
    """
    Model Exponential Moving Average (EMA).
    Empirically improves generalization and stability in noisy datasets like MABe.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in msd.items():
            if k in esd:
                esd[k].copy_(self.decay * esd[k] + (1.0 - self.decay) * v)

    def apply_to(self, model_to_verify=None):
        """
        Copy EMA weights to another model (usually the main model during validation).
        """
        if model_to_verify is None:
            model_to_verify = self.model
        model_to_verify.load_state_dict(self.ema_model.state_dict())
