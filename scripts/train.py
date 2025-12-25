import yaml
import torch
import os
import sys

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

def train():
    # 1. Load Config
    config_path = 'configs/base_config.yaml'
    if not os.path.exists(config_path):
        # Fallback for running from root
        config_path = 'configs/base_config.yaml'
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Starting Experiment: {config['experiment_name']}")
    
    # 2. Initialize Data Module (Module I)
    # Note: In a real run, you would split data into train/val based on video_ids
    # Here we simulate with the same dataset for demo
    train_dataset = MABeDataset(config['data']['data_dir'], config, mode='train')
    val_dataset = MABeDataset(config['data']['data_dir'], config, mode='val')
    
    # Collate function might be needed if time lengths vary (due to augmentation)
    # For now assuming fixed length or batch size 1 for simplicity if variable
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    # 3. Initialize Model (Module II & III)
    # Dynamically calculate input_dim based on features
    # Assuming 2 mice and 7 keypoints if unification is enabled, else 5 (or whatever baseline)
    num_mice = 2
    
    if config['data']['preprocessing'].get('unify_body_parts', False):
        num_keypoints = 7 # Unified 7-point skeleton
        print("Using Unified 7-point Skeleton")
    else:
        # Fallback or raw mode - this might be risky if data is mixed
        # Assuming baseline 5 points for now if disabled
        num_keypoints = 5 
        print("Using Baseline/Raw Skeleton (Warning: Ensure data consistency)")
    
    # Get feature dim from generator
    input_dim = train_dataset.feature_generator.get_feature_dim(num_mice, num_keypoints)
    config['model']['temporal_backbone']['input_dim'] = input_dim
    print(f"Calculated Input Dimension: {input_dim}")
    
    # Update num_classes from dataset
    config['model']['classifier']['num_classes'] = train_dataset.num_classes
    print(f"Detected {train_dataset.num_classes} classes from dataset.")
    
    model = HHSTFModel(config['model'])
    
    # 4. Initialize Post-Processor (Module IV)
    post_processor = PostProcessor(config['post_processing'])
    
    # 5. Initialize Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    
    # 6. Training Loop
    for epoch in range(config['training']['epochs']):
        train_loss = trainer.train_epoch(epoch)
        val_f1, _, _ = trainer.validate(epoch, post_processor)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val F1 = {val_f1:.4f}")
        
        trainer.save_results(epoch, val_f1)
        trainer.scheduler.step()
        
    print("Training completed.")

if __name__ == '__main__':
    train()
