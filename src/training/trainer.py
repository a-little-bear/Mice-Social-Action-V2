import torch
import torch.nn as nn
import numpy as np
import os
import json
import inspect
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict
from .losses import FocalLoss, MacroSoftF1Loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda', feature_generator=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.feature_generator = feature_generator
        if self.feature_generator:
            # No need to move to device yet, will be done in train_epoch if needed
            # but actually FeatureGenerator is just a collection of torch ops
            pass
        
        # Check for fused AdamW support
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused else dict()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training'].get('weight_decay', 0.01)),
            **extra_args
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs']
        )
        
        loss_type = config['training']['loss_type']
        if loss_type == 'softmax':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(reduction='none')
        elif loss_type == 'soft_f1':
            self.criterion = MacroSoftF1Loss(num_classes=37)
        elif loss_type == 'bce':
            pos_weight_val = float(config['training'].get('pos_weight', 1.0))
            # Create a tensor for pos_weight to broadcast correctly
            # We don't know num_classes here easily without passing it, but BCE handles scalar broadcasting usually?
            # Actually BCEWithLogitsLoss pos_weight must be a vector of length num_classes or broadcastable.
            # If we pass a scalar tensor, it broadcasts.
            pos_weight = torch.tensor(pos_weight_val, device=device)
            self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        self.best_f1 = 0.0
        self.results = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        # Debug: Print stats for the first batch
        first_batch = True
        
        for batch in pbar:
            keypoints, labels, lab_ids, subject_ids, video_ids = batch
            
            keypoints = keypoints.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if first_batch:
                print(f"\n[DEBUG Epoch {epoch} Batch 0]")
                print(f"  Labels Shape: {labels.shape}")
                print(f"  Labels NaNs: {torch.isnan(labels).sum().item()}")
                print(f"  Labels Positives: {torch.nansum(labels).item()}")
                print(f"  Labels Mean (ignoring NaNs): {torch.nanmean(labels).item()}")
                first_batch = False
            
            # Generate features on GPU
            if self.feature_generator:
                features = self.feature_generator(keypoints)
            else:
                features = keypoints
            
            # Convert lab_ids and subject_ids to tensors if they aren't already
            # This avoids graph breaks in torch.compile
            if not isinstance(lab_ids, torch.Tensor):
                model_to_check = self.model
                if hasattr(model_to_check, '_orig_mod'):
                    model_to_check = model_to_check._orig_mod
                
                if hasattr(model_to_check, 'context_adapter') and model_to_check.context_adapter is not None:
                    lab_map = model_to_check.context_adapter.lab_map
                    lab_indices = [lab_map.get(l, 21) for l in lab_ids]
                    lab_ids = torch.tensor(lab_indices, device=self.device)
            else:
                lab_ids = lab_ids.to(self.device, non_blocking=True)
                
            if not isinstance(subject_ids, torch.Tensor):
                subject_ids = torch.tensor(subject_ids, device=self.device)
            else:
                subject_ids = subject_ids.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(features, lab_ids, subject_ids)
                
                # Handle NaNs for target generation to avoid issues in compiled kernels
                labels_fixed = torch.nan_to_num(labels, nan=0.0)
                
                if isinstance(outputs, tuple):
                    logits, detection_logits = outputs
                    # Debug logits
                    if epoch == 0 and torch.rand(1).item() < 0.01:
                         print(f"  Logits Mean: {logits.mean().item()}, Std: {logits.std().item()}")
                         
                    detection_targets = (labels_fixed.sum(dim=-1) > 0).float().unsqueeze(-1)
                    det_loss = nn.BCEWithLogitsLoss()(detection_logits, detection_targets)
                    
                    if self.config['training']['loss_type'] == 'softmax':
                        targets = torch.argmax(labels_fixed, dim=-1)
                        cls_loss = self.criterion(logits.transpose(1, 2), targets)
                    else:
                        cls_loss = self.criterion(logits, labels_fixed)
                    
                    loss = cls_loss.mean() + 0.5 * det_loss
                else:
                    logits = outputs
                    if self.config['training']['loss_type'] == 'softmax':
                        targets = torch.argmax(labels_fixed, dim=-1)
                        loss = self.criterion(logits.transpose(1, 2), targets)
                    elif self.config['training']['loss_type'] == 'soft_f1':
                        loss = self.criterion(logits, labels_fixed)
                    else:
                        loss = self.criterion(logits, labels_fixed)
                
                if self.config['training']['mask_unlabeled'] and self.config['training']['loss_type'] not in ['soft_f1']:
                    mask = ~torch.isnan(labels).any(dim=-1) if labels.ndim == 3 else ~torch.isnan(labels)
                    if loss.ndim > 0:
                        # Ensure mask matches loss dimensions for broadcasting
                        while mask.ndim < loss.ndim:
                            mask = mask.unsqueeze(-1)
                        
                        # Use torch.where for more stable compilation of masked loss
                        loss = torch.where(mask, loss, torch.zeros_like(loss))
                        # Calculate denominator based on actual masked elements
                        denom = mask.expand_as(loss).sum()
                        loss = loss.sum() / (denom + 1e-6)
                elif loss.ndim > 0:
                    loss = loss.mean()
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch, post_processor=None):
        self.model.eval()
        
        # Incremental stats storage: lab_id -> {class_idx: {'tp': 0, 'fp': 0, 'fn': 0}}
        lab_stats = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                keypoints, labels, lab_ids, subject_ids, video_ids = batch
                
                keypoints = keypoints.to(self.device)
                
                # Generate features on GPU
                if self.feature_generator:
                    features = self.feature_generator(keypoints)
                else:
                    features = keypoints
                
                # Convert lab_ids and subject_ids to tensors for the model
                if not isinstance(lab_ids, torch.Tensor):
                    model_to_check = self.model
                    if hasattr(model_to_check, '_orig_mod'):
                        model_to_check = model_to_check._orig_mod
                    
                    if hasattr(model_to_check, 'context_adapter') and model_to_check.context_adapter is not None:
                        lab_map = model_to_check.context_adapter.lab_map
                        lab_indices = [lab_map.get(l, 21) for l in lab_ids]
                        lab_ids_dev = torch.tensor(lab_indices, device=self.device)
                    else:
                        lab_ids_dev = lab_ids
                else:
                    lab_ids_dev = lab_ids.to(self.device)
                    
                if not isinstance(subject_ids, torch.Tensor):
                    subject_ids_dev = torch.tensor(subject_ids, device=self.device)
                else:
                    subject_ids_dev = subject_ids.to(self.device)
                
                outputs = self.model(features, lab_ids_dev, subject_ids_dev)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                    
                probs = torch.sigmoid(logits)
                
                # Binarize (Simple thresholding for validation monitoring to save memory)
                # Note: Post-processing (smoothing) is skipped here because validation batches 
                # are shuffled windows, so temporal smoothing across windows is invalid.
                preds_bin = (probs > 0.5).float()
                
                # Move to CPU for stats accumulation
                preds_np = preds_bin.cpu().numpy()
                targets_np = labels.cpu().numpy()
                
                # Handle lab_ids being a tuple or list from DataLoader
                if isinstance(lab_ids, torch.Tensor):
                    lab_ids_np = lab_ids.cpu().numpy()
                else:
                    lab_ids_np = np.array(lab_ids)

                # Update stats incrementally
                B, T, C = preds_np.shape
                
                for i in range(B):
                    lab = lab_ids_np[i]
                    
                    # Mask valid frames (not NaN in targets)
                    # targets_np[i]: [T, C]
                    valid_mask = ~np.isnan(targets_np[i]).any(axis=-1) # [T]
                    
                    if not np.any(valid_mask):
                        continue
                        
                    p = preds_np[i][valid_mask] # [T_valid, C]
                    t = targets_np[i][valid_mask] # [T_valid, C]
                    
                    for c in range(C):
                        tp = np.sum((p[:, c] == 1) & (t[:, c] == 1))
                        fp = np.sum((p[:, c] == 1) & (t[:, c] == 0))
                        fn = np.sum((p[:, c] == 0) & (t[:, c] == 1))
                        
                        lab_stats[lab][c]['tp'] += tp
                        lab_stats[lab][c]['fp'] += fp
                        lab_stats[lab][c]['fn'] += fn

        # Compute F1 from accumulated stats
        lab_scores = []
        
        for lab, class_stats in lab_stats.items():
            action_f1s = []
            # Assuming num_classes is consistent
            # We iterate over keys in class_stats which are class indices
            for c, stats in class_stats.items():
                tp = stats['tp']
                fp = stats['fp']
                fn = stats['fn']
                
                if (2 * tp + fp + fn) == 0:
                    f1 = 0.0
                else:
                    f1 = (2 * tp) / (2 * tp + fp + fn)
                action_f1s.append(f1)
            
            if action_f1s:
                lab_scores.append(np.mean(action_f1s))
        
        final_f1 = np.mean(lab_scores) if lab_scores else 0.0
        
        # Return None for preds/targets to save memory
        return final_f1, None, None

    def save_results(self, epoch, f1):
        self.results.append({'epoch': epoch, 'f1': f1})
        
        save_dir = self.config['output']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
            
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"New best model saved with F1: {f1:.4f}")
