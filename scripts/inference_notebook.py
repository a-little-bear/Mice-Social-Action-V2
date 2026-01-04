
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fusion_model import HHSTFModel
from src.data.dataset import MABeDataset
from src.postprocessing.optimization import PostProcessor
from src.postprocessing.notebook_logic import TT_PER_LAB_NN

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = 'configs/base_config.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return
    config = load_config(config_path)
    
    config['data']['preload'] = False 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Test Data...")
    test_dataset = MABeDataset(config['data']['data_dir'], config, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print("Loading Model...")
    model = HHSTFModel(config['model'], feature_generator=test_dataset.feature_generator)
    
    checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    print("Running Inference...")
    all_preds = []
    all_meta = []
    
    actions = test_dataset.classes
    if not actions:
        print("Warning: Could not load classes from dataset. Using hardcoded list.")
        actions = sorted(list(set(a for lab in TT_PER_LAB_NN.values() for a in lab.keys())))
    
    print("Applying Notebook Post-Processing...")
    pp = PostProcessor(config['post_processing'])
    
    print("Done! Please adapt the data collection loop to your specific dataset output format.")

if __name__ == '__main__':
    main()
