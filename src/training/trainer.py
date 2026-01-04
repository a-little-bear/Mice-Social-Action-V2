import torch
import torch.nn as nn
import numpy as np
import os
import json
import inspect
import gc
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict
from .losses import FocalLoss, MacroSoftF1Loss, OHEMLoss
from ..postprocessing.optimization import PostProcessor

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda', feature_generator=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.feature_generator = feature_generator
        self.post_processor = PostProcessor(config['post_processing'])
        
        if self.feature_generator:
            pass
        
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
        num_classes = config['model']['classifier']['num_classes']
        
        if loss_type == 'softmax':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss_type == 'focal':
            pos_weight_val = float(config['training'].get('pos_weight', 1.0))
            pos_weight = torch.tensor(pos_weight_val, device=device)
            gamma = float(config['training'].get('focal_gamma', 2.0))
            alpha = float(config['training'].get('focal_alpha', 0.25))
            self.criterion = FocalLoss(reduction='none', pos_weight=pos_weight, gamma=gamma, alpha=alpha)
        elif loss_type == 'soft_f1':
            self.criterion = MacroSoftF1Loss(num_classes=num_classes)
        elif loss_type == 'macro_soft_f1':
            self.criterion = MacroSoftF1Loss(num_classes=num_classes)
        elif loss_type == 'ohem':
            rate = config['training'].get('ohem_percent', 0.7)
            self.criterion = OHEMLoss(rate=rate)
        elif loss_type == 'bce':
            pos_weight_val = float(config['training'].get('pos_weight', 1.0))
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
        
        first_batch = True
        
        for batch in pbar:
            keypoints, labels, lab_ids, subject_ids, video_ids = batch
            
            keypoints = keypoints.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            labels_fixed = torch.nan_to_num(labels, nan=0.0)
            
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
            
            if self.config['training'].get('dynamic_pos_weight', False):
                pos = labels_fixed.sum(dim=(0, 1))
                neg = (1.0 - labels_fixed).sum(dim=(0, 1))
                new_pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=1000.0)
                
                if hasattr(self.criterion, 'pos_weight'):
                    self.criterion.pos_weight = new_pos_weight
                elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=new_pos_weight)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(keypoints, lab_ids, subject_ids)
                
                mask = None
                if self.config['training']['mask_unlabeled']:
                    mask = ~torch.isnan(labels).any(dim=-1) if labels.ndim == 3 else ~torch.isnan(labels)

                if isinstance(outputs, tuple):
                    logits, detection_logits = outputs
                         
                    detection_targets = (labels_fixed.sum(dim=-1) > 0).float().unsqueeze(-1)
                    det_loss = nn.BCEWithLogitsLoss()(detection_logits, detection_targets)
                    
                    if self.config['training']['loss_type'] == 'softmax':
                        targets = torch.argmax(labels_fixed, dim=-1)
                        cls_loss = self.criterion(logits.transpose(1, 2), targets)
                    elif isinstance(self.criterion, OHEMLoss):
                        cls_loss = self.criterion(logits, labels_fixed, mask=mask)
                    else:
                        cls_loss = self.criterion(logits, labels_fixed)
                    
                    if cls_loss.ndim > 0 and mask is not None and self.config['training']['loss_type'] not in ['soft_f1', 'softmax']:
                         while mask.ndim < cls_loss.ndim:
                            mask = mask.unsqueeze(-1)
                         cls_loss = torch.where(mask, cls_loss, torch.zeros_like(cls_loss))
                         denom = mask.expand_as(cls_loss).sum()
                         cls_loss = cls_loss.sum() / (denom + 1e-6)
                    elif cls_loss.ndim > 0:
                         cls_loss = cls_loss.mean()

                    loss = cls_loss + 0.5 * det_loss
                else:
                    logits = outputs
                    if self.config['training']['loss_type'] == 'softmax':
                        targets = torch.argmax(labels_fixed, dim=-1)
                        loss = self.criterion(logits.transpose(1, 2), targets)
                    elif isinstance(self.criterion, OHEMLoss):
                        loss = self.criterion(logits, labels_fixed, mask=mask)
                    elif self.config['training']['loss_type'] == 'soft_f1':
                        loss = self.criterion(logits, labels_fixed)
                    else:
                        loss = self.criterion(logits, labels_fixed)
                
                if self.config['training']['mask_unlabeled'] and self.config['training']['loss_type'] not in ['soft_f1', 'ohem']:
                    if loss.ndim > 0:
                        while mask.ndim < loss.ndim:
                            mask = mask.unsqueeze(-1)
                        
                        loss = torch.where(mask, loss, torch.zeros_like(loss))
                        denom = mask.expand_as(loss).sum()
                        loss = loss.sum() / (denom + 1e-6)
                elif loss.ndim > 0:
                    loss = loss.mean()
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch, post_processor=None):
        if post_processor is not None:
            self.post_processor = post_processor

        self.model.eval()
        lab_stats = defaultdict(lambda: None)
        
        all_probs = []
        all_targets = []
        all_lab_ids = []
        
        eval_interval = self.config['training'].get('eval_interval', 10)
        is_full_eval_epoch = (epoch + 1) % eval_interval == 0 or (epoch + 1) == self.config['training']['epochs']
        
        collect_all = is_full_eval_epoch and (
            self.config['post_processing'].get('optimize_thresholds', False) or \
            self.config['post_processing'].get('tie_breaking', 'none') != 'none'
        )
        
        with torch.no_grad():
            if collect_all and self.config['post_processing'].get('smoothing', {}).get('method', 'none') != 'none':
                print(f"Note: Applying temporal smoothing per batch during validation to save memory.")
                
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                keypoints, labels, lab_ids, subject_ids, video_ids = batch
                
                keypoints = keypoints.to(self.device)
                
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
                
                outputs = self.model(keypoints, lab_ids_dev, subject_ids_dev)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                    
                probs = torch.sigmoid(logits)
                
                if collect_all:
                    all_probs.append(probs.half().cpu())
                    all_targets.append(labels.half().cpu())
                    
                    if isinstance(lab_ids, torch.Tensor):
                        all_lab_ids.append(lab_ids.cpu())
                    else:
                        all_lab_ids.extend(lab_ids)
                else:
                    preds_bin = (probs > 0.5).float()
                    preds_np = preds_bin.cpu().numpy()
                    targets_np = labels.cpu().numpy()
                    
                    if isinstance(lab_ids, torch.Tensor):
                        lab_ids_np = lab_ids.cpu().numpy()
                    else:
                        lab_ids_np = np.array(lab_ids)

                    B, T, C = preds_np.shape
                    
                    unique_labs = np.unique(lab_ids_np)
                    for lab in unique_labs:
                        batch_mask = (lab_ids_np == lab)
                        
                        p_lab = preds_np[batch_mask]
                        t_lab = targets_np[batch_mask]
                        
                        valid_mask = ~np.isnan(t_lab).any(axis=-1)
                        
                        if not np.any(valid_mask):
                            continue
                            
                        p_flat = p_lab[valid_mask]
                        t_flat = t_lab[valid_mask]
                        
                        tp = (p_flat * t_flat).sum(axis=0)
                        
                        fp = (p_flat * (1 - t_flat)).sum(axis=0)
                        
                        fn = ((1 - p_flat) * t_flat).sum(axis=0)
                        
                        if lab_stats[lab] is None:
                            lab_stats[lab] = np.zeros((C, 3), dtype=np.int64)
                        
                        lab_stats[lab][:, 0] += tp.astype(np.int64)
                        lab_stats[lab][:, 1] += fp.astype(np.int64)
                        lab_stats[lab][:, 2] += fn.astype(np.int64)

        if collect_all:
            print("\n[Post-Processing] Concatenating predictions (Memory-efficient)...")
            
            N_total = sum(p.shape[0] for p in all_probs)
            T, C = all_probs[0].shape[1:]
            
            full_probs = np.empty((N_total, T, C), dtype=np.float16)
            full_targets = np.empty((N_total, T, C), dtype=np.float16)
            
            curr = 0
            while all_probs:
                p = all_probs.pop(0)
                t = all_targets.pop(0)
                batch_n = p.shape[0]
                full_probs[curr:curr+batch_n] = p.numpy()
                full_targets[curr:curr+batch_n] = t.numpy()
                curr += batch_n
                
            del all_probs
            del all_targets
            gc.collect()
            
            if self.config['post_processing'].get('smoothing', {}).get('method', 'none') != 'none':
                print(f"[Post-Processing] Applying smoothing to full validation set...")
                full_probs = self.post_processor.apply_smoothing(full_probs, verbose=True)

            flat_probs_view = full_probs.reshape(-1, C)
            flat_targets_view = full_targets.reshape(-1, C)
            
            if len(all_lab_ids) > 0 and isinstance(all_lab_ids[0], torch.Tensor):
                full_lab_ids = torch.cat(all_lab_ids, dim=0).numpy()
            else:
                full_lab_ids = np.array(all_lab_ids)
            
            flat_lab_ids = np.repeat(full_lab_ids, T)
            del full_lab_ids
            
            print("[Post-Processing] Masking invalid frames...")
            valid_mask = ~np.isnan(flat_targets_view).any(axis=-1)
            
            flat_probs = flat_probs_view[valid_mask]
            del full_probs
            gc.collect()
            
            flat_targets = flat_targets_view[valid_mask]
            del full_targets
            gc.collect()
            
            flat_lab_ids = flat_lab_ids[valid_mask]
            gc.collect()
            
            classes = getattr(self.train_loader.dataset, 'classes', None)
            self.post_processor.optimize_thresholds(flat_probs, flat_targets, flat_lab_ids, classes=classes)
            
            if self.config['post_processing'].get('tie_breaking', 'none') != 'none':
                final_preds = self.post_processor.apply_tie_breaking(flat_probs, flat_lab_ids)
            else:
                final_preds = self.post_processor.apply_thresholds(flat_probs, flat_lab_ids)
            
            results = self.post_processor.calculate_f1_scores(final_preds, flat_targets, flat_lab_ids)
            final_f1 = results['overall']
            
            del flat_probs, flat_targets, flat_lab_ids, final_preds
            gc.collect()
            torch.cuda.empty_cache()
            
            return final_f1, None, None

        lab_scores = []
        for lab, stats in lab_stats.items():
            if stats is None:
                continue
            tp = stats[:, 0]
            fp = stats[:, 1]
            fn = stats[:, 2]
            denom = 2 * tp + fp + fn
            f1s = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
            
            positives = tp + fn
            present_classes = (positives > 0)
            
            if present_classes.sum() == 0:
                lab_scores.append(0.0)
            else:
                lab_scores.append(np.mean(f1s[present_classes]))
        
        final_f1 = np.mean(lab_scores) if lab_scores else 0.0
        
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
