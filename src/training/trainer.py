import torch
import torch.nn as nn
import numpy as np
import os
import json
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
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate']
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
            features, labels, lab_ids, subject_ids = batch
            
            features = features.to(self.device)
            labels = labels.to(self.device)
            if isinstance(lab_ids, torch.Tensor):
                lab_ids = lab_ids.to(self.device)
            if isinstance(subject_ids, torch.Tensor):
                subject_ids = subject_ids.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features, lab_ids, subject_ids)
            
            if isinstance(outputs, tuple):
                logits, detection_logits = outputs
                detection_targets = (labels.sum(dim=-1) > 0).float().unsqueeze(-1)
                det_loss = nn.BCEWithLogitsLoss()(detection_logits, detection_targets)
                
                if self.config['training']['loss_type'] == 'softmax':
                    targets = torch.argmax(labels, dim=-1)
                    cls_loss = self.criterion(logits.transpose(1, 2), targets)
                else:
                    cls_loss = self.criterion(logits, labels)
                
                loss = cls_loss.mean() + 0.5 * det_loss
            else:
                logits = outputs
                if self.config['training']['loss_type'] == 'softmax':
                    targets = torch.argmax(labels, dim=-1)
                    loss = self.criterion(logits.transpose(1, 2), targets)
                elif self.config['training']['loss_type'] == 'soft_f1':
                    loss = self.criterion(logits, labels)
                else:
                    loss = self.criterion(logits, labels)
            
            if self.config['training']['mask_unlabeled'] and self.config['training']['loss_type'] not in ['soft_f1']:
                mask = ~torch.isnan(labels).any(dim=-1) if labels.ndim == 3 else ~torch.isnan(labels)
                if loss.ndim > 0:
                    if mask.ndim < loss.ndim:
                        mask = mask.unsqueeze(-1)
                    loss = loss * mask.float()
                    loss = loss.sum() / (mask.sum() * loss.shape[-1] + 1e-6)
            elif loss.ndim > 0:
                loss = loss.mean()
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch, post_processor=None):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_lab_ids = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                features, labels, lab_ids, subject_ids = batch
                
                features = features.to(self.device)
                lab_ids = lab_ids.to(self.device)
                subject_ids = subject_ids.to(self.device)
                
                outputs = self.model(features, lab_ids, subject_ids)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                    
                probs = torch.sigmoid(logits)
                
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                
                T = features.shape[1]
                lab_ids_expanded = lab_ids.unsqueeze(1).expand(-1, T)
                all_lab_ids.append(lab_ids_expanded.cpu().numpy())

        flat_preds = np.concatenate(all_preds).reshape(-1, all_preds[0].shape[-1])
        flat_targets = np.concatenate(all_targets).reshape(-1, all_targets[0].shape[-1])
        flat_labs = np.concatenate(all_lab_ids).reshape(-1)
        
        valid_mask = ~np.isnan(flat_targets).any(axis=1)
        flat_preds = flat_preds[valid_mask]
        flat_targets = flat_targets[valid_mask]
        flat_labs = flat_labs[valid_mask]
        
        if post_processor and self.config['post_processing']['optimize_thresholds']:
            post_processor.optimize_thresholds(flat_preds, flat_targets, flat_labs)
            
        if post_processor:
            binary_preds = post_processor(flat_preds, flat_labs)
        else:
            binary_preds = (flat_preds > 0.5).astype(int)
            
        f1 = f1_score(flat_targets, binary_preds, average='macro')
        
        return f1, flat_preds, flat_targets

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
