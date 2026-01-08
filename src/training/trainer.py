import torch
import torch.nn as nn
import numpy as np
import os
import json
import inspect
import gc
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict
from .losses import FocalLoss, MacroSoftF1Loss, OHEMLoss
from ..postprocessing.optimization import PostProcessor
from ..utils.ema import EMA

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
        elif loss_type == 'new_focal':
            # Extract weights from Dataset
            ds = train_loader.dataset
            # If dataset is wrapped in a Subset, we might need to go deeper (though typically it's the direct dataset)
            while hasattr(ds, 'dataset'):
                ds = ds.dataset
            
            # Get global positive weight scaling factor
            scale_factor = float(config['training'].get('pos_weight', 1.0))

            if hasattr(ds, 'class_weights'):
                # Combine relative class weights (mean~1.0) with global scaling factor
                pos_weight = ds.class_weights.to(device) * scale_factor
                print(f"Using class-aware weights for 'new_focal' scaled by {scale_factor}. Weights mean: {pos_weight.mean().item():.4f}")
            else:
                pos_weight = torch.tensor(scale_factor, device=device)
                print("Warning: 'new_focal' requested but class_weights not found. Using default.")
                
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
        
        # New: Model Weight EMA
        self.ema = None
        if config['training'].get('ema_enabled', True):
            decay = config['training'].get('ema_decay', 0.999)
            self.ema = EMA(self.model, decay=decay)
            print(f"Model EMA initialized with decay: {decay}")

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
            
            if self.ema:
                self.ema.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch, post_processor=None, force_full=False):
        if post_processor is not None:
            self.post_processor = post_processor

        self.model.eval()
        
        # New: Use EMA weights for validation if enabled
        original_state_dict = None
        if self.ema:
            # Save current weights to restore after validation
            original_state_dict = deepcopy(self.model.state_dict())
            self.ema.apply_to(self.model)
            print("Using EMA weights for validation")

        # 核心初始化：确保所有列表在任何 eval 模式下都已定义
        all_probs = []
        all_targets = []
        all_lab_ids = []
        all_video_ids = [] 
        
        eval_interval = self.config['training'].get('eval_interval', 10)
        is_full_eval_epoch = force_full or (epoch + 1) % eval_interval == 0 or (epoch + 1) == self.config['training']['epochs']
        
        collect_all = is_full_eval_epoch and (
            self.config['post_processing'].get('optimize_thresholds', False) or \
            self.config['post_processing'].get('tie_breaking', 'none') != 'none'
        )
        
        # 性能极速优化：在 GPU 上完成所有指标累加，彻底消除 CPU 瓶颈
        ds = self.val_loader.dataset
        unique_labs = sorted(list(set(d['lab_id'] for d in ds.data)))
        lab_to_idx = {lab: i for i, lab in enumerate(unique_labs)}
        num_labs = len(unique_labs)
        num_classes = ds.num_classes
        
        # [num_labs, num_classes, 3] -> 0:TP, 1:FP, 2:FN
        gpu_stats = torch.zeros((num_labs, num_classes, 3), device=self.device, dtype=torch.float64)

        with torch.no_grad():
            if collect_all and self.config['post_processing'].get('smoothing', {}).get('method', 'none') != 'none':
                print(f"Note: Applying temporal smoothing per batch during validation to save memory.")
                
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                keypoints, labels, lab_ids, subject_ids, video_ids = batch
                
                keypoints = keypoints.to(self.device)
                labels = labels.to(self.device).float() # 统一到 float 用于计算
                
                # 获取 Lab 索引 (GPU Tensor)
                lab_indices = [lab_to_idx.get(l, 0) for l in lab_ids]
                batch_lab_indices = torch.tensor(lab_indices, device=self.device)
                
                if not isinstance(subject_ids, torch.Tensor):
                    subject_ids_dev = torch.tensor(subject_ids, device=self.device)
                else:
                    subject_ids_dev = subject_ids.to(self.device)
                
                # 重新计算 lab_ids 到模型需要的索引（如果适用）
                model_to_check = self.model
                if hasattr(model_to_check, '_orig_mod'): model_to_check = model_to_check._orig_mod
                
                if hasattr(model_to_check, 'context_adapter') and model_to_check.context_adapter is not None:
                    m_lab_map = model_to_check.context_adapter.lab_map
                    m_lab_indices = [m_lab_map.get(l, 21) for l in lab_ids]
                    lab_ids_dev = torch.tensor(m_lab_indices, device=self.device)
                else:
                    lab_ids_dev = batch_lab_indices 

                outputs = self.model(keypoints, lab_ids_dev, subject_ids_dev)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                    
                probs = torch.sigmoid(logits)
                
                if collect_all:
                    all_probs.append(probs.half().cpu())
                    all_targets.append(labels.byte().cpu())
                    all_lab_ids.extend(lab_ids)
                    all_video_ids.extend(video_ids)
                else:
                    preds_bin = (probs > 0.5).float()
                    
                    # 矢量化对齐官方 Active Masking
                    video_masks = getattr(ds, 'video_masks', None)
                    v_id_to_int = getattr(ds, 'video_id_to_int', None)
                    
                    if video_masks is not None and v_id_to_int is not None:
                        v_indices = [v_id_to_int.get(str(vid), -1) for vid in video_ids]
                        # 仅在所有视频都包含在掩码中时运行
                        if -1 not in v_indices:
                            batch_masks = video_masks[v_indices].to(self.device).unsqueeze(1)
                            preds_bin = preds_bin * batch_masks 

                    # GPU 上的 TP/FP/FN 批量计算
                    # [B, T, C] -> 对 T 维度求和 -> [B, C]
                    tp_b = (preds_bin * labels).sum(dim=1).double()
                    fp_b = (preds_bin * (1 - labels)).sum(dim=1).double()
                    fn_b = ((1 - preds_bin) * labels).sum(dim=1).double()
                    
                    # 使用 scatter_add 将 Batch 里的结果按 Lab 归位
                    # indices_expanded: [B, C]
                    indices_expanded = batch_lab_indices.view(-1, 1).expand(-1, num_classes)
                    gpu_stats[:, :, 0].scatter_add_(0, indices_expanded, tp_b)
                    gpu_stats[:, :, 1].scatter_add_(0, indices_expanded, fp_b)
                    gpu_stats[:, :, 2].scatter_add_(0, indices_expanded, fn_b)

        if not collect_all:
            # 汇总结果
            cpu_stats = gpu_stats.cpu().numpy()
            lab_scores = []
            for i, lab in enumerate(unique_labs):
                tp, fp, fn = cpu_stats[i, :, 0], cpu_stats[i, :, 1], cpu_stats[i, :, 2]
                denom = 2 * tp + fp + fn
                f1s = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
                
                # 只对在该 Lab 中出现过的类别求平均（对齐官方 Macro F1 逻辑）
                present = (tp + fn) > 0
                if present.any():
                    lab_scores.append(np.mean(f1s[present]))
                else:
                    lab_scores.append(0.0)
            
            final_f1 = np.mean(lab_scores) if lab_scores else 0.0
            
            # 恢复 EMA（如果适用）
            if original_state_dict:
                self.model.load_state_dict(original_state_dict)
                
            return final_f1, None, None

        if collect_all:
            print("\n[Post-Processing] Concatenating predictions (Memory-efficient)...")
            
            N_total = sum(p.shape[0] for p in all_probs)
            T, C = all_probs[0].shape[1:]
            
            # 【优化】使用 float16 和 uint8 显著降低内存占用
            # 先分配内存，然后逐个填充并释放源 Tensor
            full_probs = np.empty((N_total, T, C), dtype=np.float16)
            full_targets = np.empty((N_total, T, C), dtype=np.uint8)
            
            curr = 0
            while all_probs:
                p = all_probs.pop(0)
                t = all_targets.pop(0)
                batch_n = p.shape[0]
                full_probs[curr:curr+batch_n] = p.numpy().astype(np.float16)
                full_targets[curr:curr+batch_n] = t.numpy().astype(np.uint8)
                curr += batch_n
                # 显式释放以帮助 GC
                del p, t
                
            gc.collect()
            
            if self.config['post_processing'].get('smoothing', {}).get('method', 'none') != 'none':
                # full_probs 已经是 float16，内部 apply_smoothing 会分块转 float32 处理
                full_probs = self.post_processor.apply_smoothing(full_probs, verbose=True)

            flat_probs_view = full_probs.reshape(-1, C)
            flat_targets_view = full_targets.reshape(-1, C)
            
            # 及时释放不再需要的维度
            # del full_probs, full_targets # 注意：flat_probs_view 是一个 view，不能删掉原对象
            
            if len(all_lab_ids) > 0 and isinstance(all_lab_ids[0], torch.Tensor):
                full_lab_ids = torch.cat(all_lab_ids, dim=0).numpy()
            else:
                full_lab_ids = np.array(all_lab_ids)
            
            flat_lab_ids = np.repeat(full_lab_ids, T)
            del full_lab_ids
            all_lab_ids = [] # 清空列表
            
            print("[Post-Processing] Masking invalid frames...")
            # 优化：如果是 uint8 且 0 是填充，我们需要确定有效范围
            # 这里暂时维持原逻辑，但加上下采样开关以应对内存极限
            
            # 由于 flat_targets_view 是 uint8，np.isnan 不适用。
            # 我们假设 targets 里的 NaN 在 loader 中被处理为了 -1 或 0。
            # 实际上在 collate_fn 中我们用的是 zeros。
            # 正确的做法是基于真实长度，或者检查 targets 是否全为 0 且 label 包含 None。
            # 考虑到 MABe 的 background 是 0，简单检查 any() 可能误删。
            # 暂且保留 valid_mask，但修正 uint8 的检测方式
            if flat_targets_view.dtype == np.uint8:
                # uint8 不会有 NaN，如果存在 masking，通常用 255
                valid_mask = np.ones(len(flat_targets_view), dtype=bool)
            else:
                valid_mask = ~np.isnan(flat_targets_view).any(axis=-1)
            
            # 由于全量后处理需要对齐视频，我们需要重复 video_ids 以构建 flat_video_ids
            flat_video_ids = np.repeat(all_video_ids, T)
            
            # 【内存优化】下采样进行阈值优化，但在 F1 计算时使用全量
            optimize_stride = 1
            if flat_probs_view.nbytes > 15 * 1024 * 1024 * 1024:
                print(f"Large prediction set ({flat_probs_view.nbytes/1e9:.2f} GB). Downsampling 5x for threshold search.")
                optimize_stride = 5

            flat_probs_opt = flat_probs_view[::optimize_stride]
            flat_targets_opt = flat_targets_view[::optimize_stride]
            flat_lab_ids_opt = flat_lab_ids[::optimize_stride]
            
            classes = getattr(self.val_loader.dataset, 'classes', None)
            video_to_active_indices = getattr(self.val_loader.dataset, 'video_to_active_indices', None)
            
            self.post_processor.optimize_thresholds(
                flat_probs_opt, flat_targets_opt, flat_lab_ids_opt, 
                classes=classes, 
                video_to_active_indices=video_to_active_indices
            )
            
            del flat_probs_opt, flat_targets_opt, flat_lab_ids_opt
            gc.collect()

            final_bin_preds = self.post_processor.apply_tie_breaking(flat_probs_view, flat_lab_ids)
            
            # 计算最终指标时应用 Active Label Masking
            v_id_to_mask = getattr(self.val_loader.dataset, 'video_masks', None)
            v_id_to_int = getattr(self.val_loader.dataset, 'video_id_to_int', None)
            
            # 使用列表解析构建 flat_video_ids 以便传递给 calculate_f1_scores
            flat_video_ids = np.repeat(all_video_ids, T)
            
            if v_id_to_mask is not None and v_id_to_int is not None:
                print("[Post-Processing] Applying Active Label Masking to final predictions...")
                # 批量查找视频索引
                v_indices = [v_id_to_int.get(str(vid), -1) for vid in all_video_ids]
                # 重复 T 次
                flat_v_indices = np.repeat(v_indices, T)
                # 获取掩码 (CPU)
                masks_cpu = v_id_to_mask.numpy()
                final_masks = masks_cpu[flat_v_indices]
                # 原地屏蔽
                final_bin_preds &= final_masks.astype(np.uint8)
                del final_masks
                
            # 使用 post_processor 计算 F1
            results = self.post_processor.calculate_f1_scores(
                final_bin_preds, flat_targets_view, flat_lab_ids,
                video_to_active_indices=video_to_active_indices,
                flat_video_ids=flat_video_ids
            )
            final_f1 = results['overall']
            
            del final_bin_preds, full_probs, full_targets, flat_probs_view, flat_targets_view, flat_video_ids
            gc.collect()
            torch.cuda.empty_cache()
            
            # Restore original weights if EMA was used
            if original_state_dict:
                self.model.load_state_dict(original_state_dict)
                
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
        
        # Restore original weights if EMA was used
        if original_state_dict:
            self.model.load_state_dict(original_state_dict)
            del original_state_dict
            
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
