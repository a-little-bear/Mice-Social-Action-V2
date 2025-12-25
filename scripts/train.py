import yaml
import torch
import os
import sys
import numpy as np
import gc
import atexit
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

def cleanup():
    """Cleanup function to prevent memory leaks on crash or exit."""
    print("\nPerforming memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear torch.compile artifacts if any
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
        
    print("Cleanup complete.")

# Register cleanup to run on exit
atexit.register(cleanup)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MABeDataset
from src.models.fusion_model import HHSTFModel
from src.postprocessing.optimization import PostProcessor
from src.training.trainer import Trainer

def collate_fn(batch):
    keypoints, labels, lab_ids, subject_ids, video_ids = zip(*batch)
    lengths = [k.shape[0] for k in keypoints]
    max_len = max(lengths)
    
    # keypoints[0] shape: (T, M, K, 2)
    kp_shape = keypoints[0].shape[1:]
    padded_keypoints = torch.zeros(len(keypoints), max_len, *kp_shape)
    padded_labels = torch.zeros(len(labels), max_len, labels[0].shape[1])
    
    for i, (k, l) in enumerate(zip(keypoints, labels)):
        end = lengths[i]
        padded_keypoints[i, :end] = k
        padded_labels[i, :end] = l
        
    return padded_keypoints, padded_labels, lab_ids, subject_ids, video_ids

def run_fold(config, train_loader, val_loader, device, fold_idx=None, input_dim=None, num_classes=None, feature_generator=None):
    print(f"Initializing model for {'Fold ' + str(fold_idx) if fold_idx is not None else 'Single Run'}...")
    
    # Update config with dynamic dims if provided
    if input_dim:
        config['model']['temporal_backbone']['input_dim'] = input_dim
    if num_classes:
        config['model']['classifier']['num_classes'] = num_classes

    model = HHSTFModel(config['model'])
    model = model.to(device)

    # Optimization: torch.compile
    if not config.get('test', False) and config['training'].get('torch_compile', True):
        try:
            print("Compiling model with torch.compile...")
            # Use reduce-overhead for better stability than max-autotune
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Compilation failed, fallback to eager mode: {e}")
    else:
        print("Skipping torch.compile (Disabled in config or Test Mode).")
    
    post_processor = PostProcessor(config['post_processing'])
    
    trainer = Trainer(model, train_loader, val_loader, config, device=device, feature_generator=feature_generator)
    
    try:
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
    finally:
        # Cleanup model and trainer to free GPU memory
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train():
    parser = argparse.ArgumentParser(description="Train MABe Mouse Behavior Detection Model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--save_dir", type=str, help="Override save directory")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override from CLI if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir

    if config.get('test', False):
        print("!!! TEST MODE ENABLED !!!")
        config['data']['data_dir'] = 'test_data/'
        config['data']['batch_size'] = 2
        config['data']['num_workers'] = 0
        config['data']['preload'] = True
        config['training']['epochs'] = 2
        if config.get('cross_validation', {}).get('enabled', False):
            config['cross_validation']['n_folds'] = 2
        print(f"Overriding config for testing: data_dir={config['data']['data_dir']}, epochs={config['training']['epochs']}")

    print(f"Starting Experiment: {config['experiment_name']}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Initialize Data
    # For CV, we use the 'train' mode dataset as the full pool
    full_dataset = MABeDataset(config['data']['data_dir'], config, mode='train')
    
    if len(full_dataset) == 0:
        print(f"ERROR: No data found in {config['data']['data_dir']}. Please check your data path.")
        return

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
            
            train_loader = None
            val_loader = None
            
            try:
                train_subsampler = SubsetRandomSampler(train_ids)
                val_subsampler = SubsetRandomSampler(val_ids)
                
                loader_kwargs = {
                    'batch_size': config['data']['batch_size'],
                    'num_workers': config['data']['num_workers'],
                    'collate_fn': collate_fn,
                    'pin_memory': config['data'].get('pin_memory', False),
                }
                
                if loader_kwargs['num_workers'] > 0:
                    loader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
                    loader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

                train_loader = DataLoader(
                    full_dataset, 
                    sampler=train_subsampler,
                    **loader_kwargs
                )
                val_loader = DataLoader(
                    full_dataset, 
                    sampler=val_subsampler,
                    **loader_kwargs
                )
                
                best_f1 = run_fold(config, train_loader, val_loader, device, fold, input_dim, num_classes, feature_generator=full_dataset.feature_generator)
                print(f"Fold {fold} Best F1: {best_f1:.4f}")
                cv_scores.append(best_f1)
            
            except Exception as e:
                print(f"Error in Fold {fold}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Explicitly shutdown loaders and clear memory
                if train_loader is not None:
                    del train_loader
                if val_loader is not None:
                    del val_loader
                cleanup()
            
        print("\n=== CV Results ===")
        print(f"Scores: {cv_scores}")
        print(f"Average F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        
    else:
        print("Starting Standard Training (Train/Val Split)...")
        # Standard split logic
        val_dataset = MABeDataset(config['data']['data_dir'], config, mode='val')
        
        train_loader = None
        val_loader = None
        
        try:
            loader_kwargs = {
                'batch_size': config['data']['batch_size'],
                'num_workers': config['data']['num_workers'],
                'collate_fn': collate_fn,
                'pin_memory': config['data'].get('pin_memory', False),
            }
            
            if loader_kwargs['num_workers'] > 0:
                loader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
                loader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

            train_loader = DataLoader(
                full_dataset, 
                shuffle=True,
                **loader_kwargs
            )
            val_loader = DataLoader(
                val_dataset, 
                shuffle=False,
                **loader_kwargs
            )
            
            run_fold(config, train_loader, val_loader, device, None, input_dim, num_classes, feature_generator=full_dataset.feature_generator)
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if train_loader is not None:
                del train_loader
            if val_loader is not None:
                del val_loader
            cleanup()

if __name__ == '__main__':
    train()
