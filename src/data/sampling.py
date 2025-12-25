import numpy as np
import torch
import pandas as pd
from torch.utils.data import WeightedRandomSampler

class ActionRichSampler:
    def __init__(self, labels, window_size, bias_factor=0.5):
        self.labels = labels
        self.window_size = window_size
        self.bias_factor = bias_factor

    def get_sampler(self):
        weights = []
        for label in self.labels:
            if isinstance(label, (np.ndarray, torch.Tensor)):
                has_action = np.any(label > 0)
            elif isinstance(label, pd.DataFrame):
                has_action = not label.empty
            elif isinstance(label, bool):
                has_action = label
            else:
                has_action = False
            
            weight = 1.0 + self.bias_factor if has_action else 1.0
            weights.append(weight)
            
        weights = torch.DoubleTensor(weights)
        return WeightedRandomSampler(weights, len(weights))

