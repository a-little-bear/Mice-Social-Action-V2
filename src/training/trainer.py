import torch
import torch.nn as nn
import numpy as np
import os
import json
import inspect
from tqdm import tqdm
from sklearn.metrics import f1_score
from .losses import FocalLoss, MacroSoftF1Loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
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
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        self.best_f1 = 0.0
        self.results = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            features, labels, lab_ids, subject_ids, video_ids = batch
            
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
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
        all_preds = []
        all_targets = []
        all_lab_ids = []
        all_video_ids = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                features, labels, lab_ids, subject_ids, video_ids = batch
                
                features = features.to(self.device)
                
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
                
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                all_lab_ids.append(np.array(lab_ids))
                all_video_ids.append(np.array(video_ids))

        # Flatten batches but keep video/lab structure
        # preds: [N_samples, T, C], targets: [N_samples, T, C]
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        labs = np.concatenate(all_lab_ids, axis=0)
        vids = np.concatenate(all_video_ids, axis=0)
        
        num_classes = preds.shape[-1]
        
        # Apply post-processing if enabled
        if post_processor:
            # Flatten for post-processor which expects [Total_Frames, C]
            T = preds.shape[1]
            flat_preds = preds.reshape(-1, num_classes)
            flat_labs = np.repeat(labs, T)
            binary_preds = post_processor(flat_preds, flat_labs).reshape(preds.shape)
        else:
            binary_preds = (preds > 0.5).astype(int)

        # Official MABe F1 Logic: Aggregate TP, FP, FN per Lab
        lab_scores = []
        unique_labs = np.unique(labs)
        
        for lab in unique_labs:
            lab_mask = (labs == lab)
            lab_preds = binary_preds[lab_mask] # [N_lab_videos, T, C]
            lab_targets = targets[lab_mask]     # [N_lab_videos, T, C]
            
            # Filter out NaN frames (unlabeled)
            # valid_mask: [N_lab_videos, T]
            valid_mask = ~np.isnan(lab_targets).any(axis=-1)
            
            action_f1s = []
            for c in range(num_classes):
                tp = 0
                fp = 0
                fn = 0
                
                # Aggregate across all videos in this lab
                for v in range(lab_preds.shape[0]):
                    v_valid = valid_mask[v]
                    if not np.any(v_valid): continue
                    
                    v_p = lab_preds[v, v_valid, c]
                    v_t = lab_targets[v, v_valid, c]
                    
                    tp += np.sum((v_p == 1) & (v_t == 1))
                    fp += np.sum((v_p == 1) & (v_t == 0))
                    fn += np.sum((v_p == 0) & (v_t == 1))
                
                if (2 * tp + fp + fn) == 0:
                    f1 = 0.0
                else:
                    f1 = (2 * tp) / (2 * tp + fp + fn)
                action_f1s.append(f1)
            
            if action_f1s:
                lab_scores.append(np.mean(action_f1s))
        
        final_f1 = np.mean(lab_scores) if lab_scores else 0.0
        
        # For compatibility with existing save logic, return flat arrays for the best model
        return final_f1, preds.reshape(-1, num_classes), targets.reshape(-1, num_classes)

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
