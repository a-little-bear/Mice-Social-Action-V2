import torch
import numpy as np
from src.data.transforms import Augmentation

class TTAPredictor:
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.augmentor = Augmentation(config['data']['augmentation'])

    def predict(self, sample):
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = sample.unsqueeze(0).to(self.device) 
            logits_orig = self.model(input_tensor)
            probs_orig = torch.softmax(logits_orig, dim=-1).cpu().numpy()[0]

        self.model.train() 
        probs_list = [probs_orig]
        
        for _ in range(4): 
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs_list.append(probs)
                
        self.model.eval()
        probs_avg = np.mean(probs_list, axis=0)
        
        return probs_avg
