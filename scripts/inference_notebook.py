
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fusion_model import HHSTFModel
from src.data.dataset import MABeDataset
from src.postprocessing.optimization import PostProcessor
from src.postprocessing.notebook_logic import TT_PER_LAB_NN

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Config
    config_path = 'configs/base_config.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return
    config = load_config(config_path)
    
    # Override for inference
    config['data']['preload'] = False # Save RAM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Load Data
    print("Loading Test Data...")
    test_dataset = MABeDataset(config['data']['data_dir'], config, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 3. Load Model
    print("Loading Model...")
    model = HHSTFModel(config['model'], feature_generator=test_dataset.feature_generator)
    
    # Load checkpoint (Update this path to your best checkpoint)
    checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # 4. Inference Loop
    print("Running Inference...")
    all_preds = []
    all_meta = []
    
    # Get action names from dataset or config
    # Assuming dataset.classes is populated
    actions = test_dataset.classes
    if not actions:
        # Fallback if classes not loaded from train.csv
        # This might happen if train.csv is not in data_dir during test mode init
        # We can use the keys from TT_PER_LAB_NN as a superset, or hardcode
        print("Warning: Could not load classes from dataset. Using hardcoded list.")
        # Collect all unique actions from TT_PER_LAB_NN
        actions = sorted(list(set(a for lab in TT_PER_LAB_NN.values() for a in lab.keys())))
    
    # Note: This loop is a placeholder. You need to ensure your dataset returns
    # the necessary metadata (video_id, agent_id, target_id, frame) to construct the dataframe.
    # If your dataset returns windows, you need to aggregate predictions.
    
    # 5. Post-Processing (The Core Logic)
    print("Applying Notebook Post-Processing...")
    pp = PostProcessor(config['post_processing'])
    
    # Example usage:
    # df = pd.DataFrame(...) # Your predictions
    # submission = pp.apply_notebook_postprocessing(df, actions)
    # submission.to_csv("submission.csv", index=False)
    
    print("Done! Please adapt the data collection loop to your specific dataset output format.")

if __name__ == '__main__':
    main()
