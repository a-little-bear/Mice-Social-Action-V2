import yaml
import torch
import os
import sys
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MABeDataset
from src.models.fusion_model import HHSTFModel
from src.postprocessing.optimization import PostProcessor
from src.training.trainer import Trainer

def collate_fn(batch):
    features, labels, lab_ids, subject_ids = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feat_dim = features[0].shape[1]
    
    padded_features = torch.zeros(len(features), max_len, feat_dim)
    padded_labels = torch.zeros(len(labels), max_len, labels[0].shape[1])
    
    for i, (f, l) in enumerate(zip(features, labels)):
        end = lengths[i]
        if f.shape[1] != feat_dim:
             print(f"Warning: Feature dim mismatch at index {i}: {f.shape[1]} vs {feat_dim}. Truncating/Padding.")
             min_dim = min(f.shape[1], feat_dim)
             padded_features[i, :end, :min_dim] = f[:, :min_dim]
        else:
             padded_features[i, :end] = f
        padded_labels[i, :end] = l
        
    return padded_features, padded_labels, lab_ids, subject_ids

def run_fold(config, train_loader, val_loader, device, fold_idx=None, input_dim=None, num_classes=None):
    print(f"Initializing model for {'Fold ' + str(fold_idx) if fold_idx is not None else 'Single Run'}...")
    
    # Update config with dynamic dims if provided
    if input_dim:
        config['model']['temporal_backbone']['input_dim'] = input_dim
    if num_classes:
        config['model']['classifier']['num_classes'] = num_classes

    model = HHSTFModel(config['model'])
    model = model.to(device)

    # Optimization: torch.compile
    try:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"Compilation failed, fallback to eager mode: {e}")
    
    post_processor = PostProcessor(config['post_processing'])
    
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    
    best_f1 = 0.0
    for epoch in range(config['training']['epochs']):
        train_loss = trainer.train_epoch(epoch)
        val_f1, _, _ = trainer.validate(epoch, post_processor)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val F1 = {val_f1:.4f}")
        
        # Save logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_dir = config['training']['save_dir']
            if fold_idx is not None:
                save_dir = os.path.join(save_dir, f'fold_{fold_idx}')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            # Also save metrics
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                import json
                json.dump({'best_f1': best_f1, 'epoch': epoch}, f)
            
        trainer.scheduler.step()
        
    return best_f1

def train():
    # 1. Load Config
    config_path = 'configs/base_config.yaml'
    if not os.path.exists(config_path):
        config_path = 'configs/base_config.yaml'
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Starting Experiment: {config['experiment_name']}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Initialize Data
    # For CV, we use the 'train' mode dataset as the full pool
    full_dataset = MABeDataset(config['data']['data_dir'], config, mode='train')
    
    # Calculate dims once
    num_mice = 2
    if config['data']['preprocessing'].get('unify_body_parts', False):
        num_keypoints = 7
    else:
        num_keypoints = 5
        
    input_dim = full_dataset.feature_generator.get_feature_dim(num_mice, num_keypoints)
    num_classes = full_dataset.num_classes
    print(f"Input Dim: {input_dim}, Num Classes: {num_classes}")

    # Check CV config
    cv_config = config.get('cross_validation', {'enabled': False})
    
    if cv_config.get('enabled', False):
        n_folds = cv_config.get('n_folds', 5)
        print(f"Starting {n_folds}-Fold Cross Validation...")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])
        cv_scores = []
        
        # KFold expects indices or length
        indices = np.arange(len(full_dataset))
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
            print(f"\n=== FOLD {fold} ===")
            
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(
                full_dataset, 
                batch_size=config['data']['batch_size'], 
                sampler=train_subsampler,
                num_workers=config['data']['num_workers'],
                collate_fn=collate_fn,
                pin_memory=config['data'].get('pin_memory', False),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                persistent_workers=config['data'].get('persistent_workers', False)
            )
            val_loader = DataLoader(
                full_dataset, 
                batch_size=config['data']['batch_size'], 
                sampler=val_subsampler,
                num_workers=config['data']['num_workers'],
                collate_fn=collate_fn,
                pin_memory=config['data'].get('pin_memory', False),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                persistent_workers=config['data'].get('persistent_workers', False)
            )
            
            best_f1 = run_fold(config, train_loader, val_loader, device, fold, input_dim, num_classes)
            print(f"Fold {fold} Best F1: {best_f1:.4f}")
            cv_scores.append(best_f1)
            
        print("\n=== CV Results ===")
        print(f"Scores: {cv_scores}")
        print(f"Average F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        
    else:
        print("Starting Standard Training (Train/Val Split)...")
        # Standard split logic
        val_dataset = MABeDataset(config['data']['data_dir'], config, mode='val')
        
        train_loader = DataLoader(
            full_dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=True,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=config['data'].get('pin_memory', False),
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            persistent_workers=config['data'].get('persistent_workers', False)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=config['data'].get('pin_memory', False),
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            persistent_workers=config['data'].get('persistent_workers', False)
        )
        
        run_fold(config, train_loader, val_loader, device, None, input_dim, num_classes)

if __name__ == '__main__':
    train()
