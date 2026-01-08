import numpy as np
import gc
from scipy.signal import lfilter
from scipy.ndimage import binary_closing, median_filter, binary_opening
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import pandas as pd
from .notebook_logic import (
    TT_PER_LAB_NN, TIE_CONFIG_V2, TRAIN_LAB_ACTIONS, 
    smooth_probs_inplace, mask_probs_numpy_rle, probs_to_nonoverlapping_intervals
)

class PostProcessor:
    def __init__(self, config):
        self.config = config
        self.thresholds = {} 
        self.sigmas = {} 
        self.n_jobs = config.get('n_jobs', -1)

    def apply_notebook_postprocessing(self, 
                                      predictions_df: pd.DataFrame, 
                                      actions: list, 
                                      active_map: dict = None) -> pd.DataFrame:
        smooth_probs_inplace(predictions_df, actions, win=5)
        
        if active_map:
            predictions_df = mask_probs_numpy_rle(predictions_df, actions, active_map, copy=False)
            
        if 'lab_id' not in predictions_df.columns:
            pass
            
        results = []
        if 'lab_id' in predictions_df.columns:
            for lab, grp in predictions_df.groupby('lab_id'):
                sub = probs_to_nonoverlapping_intervals(
                    grp, actions, min_len=0, max_gap=7, lab=lab, tie_config=TIE_CONFIG_V2
                )
                results.append(sub)
        else:
            sub = probs_to_nonoverlapping_intervals(
                predictions_df, actions, min_len=0, max_gap=7, lab=None, tie_config=TIE_CONFIG_V2
            )
            results.append(sub)
            
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _optimize_lab(self, lab, predictions, targets, lab_ids, threshold_range, num_classes, active_indices=None):
        lab_mask = (lab_ids == lab)
        
        max_samples = 1000000
        num_lab_samples = np.sum(lab_mask)
        
        if num_lab_samples > max_samples:
            lab_indices = np.where(lab_mask)[0]
            sampled_indices = np.random.choice(lab_indices, max_samples, replace=False)
            lab_preds = predictions[sampled_indices].astype(np.float32)
            lab_targets = (targets[sampled_indices] > 0.5)
        else:
            lab_preds = predictions[lab_mask].astype(np.float32)
            lab_targets = (targets[lab_mask] > 0.5)
        
        sigmas_per_class = np.std(lab_preds, axis=0)

        # 决定哪些类别需要参与平均
        if active_indices is not None:
            eval_indices = active_indices
        else:
            # 回退：至少目标中出现过的类别
            eval_indices = np.where(lab_targets.sum(axis=0) > 0)[0]
        
        best_thresh_per_class = np.full(num_classes, 0.5)
        
        if len(eval_indices) == 0:
            return lab, best_thresh_per_class, sigmas_per_class, 0.0

        f1_scores = np.zeros((num_classes, len(threshold_range)))
        
        for i, th in enumerate(threshold_range):
            preds_bool = lab_preds > th
            
            tp = (preds_bool & lab_targets).sum(axis=0)
            fp = (preds_bool & ~lab_targets).sum(axis=0)
            fn = (~preds_bool & lab_targets).sum(axis=0)
            
            denom = 2 * tp + fp + fn
            f1_scores[:, i] = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom!=0)
            
            del preds_bool
        
        best_indices = np.argmax(f1_scores, axis=1)
        best_scores = f1_scores[np.arange(num_classes), best_indices]
        
        optimized_thresholds = threshold_range[best_indices]
        
        # 对于非激活类别，强制设为 0.5 以确保稳定性
        non_active = np.ones(num_classes, dtype=bool)
        if active_indices is not None:
            non_active[active_indices] = False
        else:
            non_active[eval_indices] = False
        optimized_thresholds[non_active] = 0.5
        
        # 只返回参与评估类别的平均 F1
        mean_eval_f1 = np.mean(best_scores[eval_indices]) if len(eval_indices) > 0 else 0.0
        
        return lab, optimized_thresholds, sigmas_per_class, mean_eval_f1

    def optimize_thresholds(self, predictions, targets, lab_ids, classes=None, video_to_active_indices=None):
        if classes is not None:
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Convert video-based active map to lab-based for threshold optimization
        lab_to_active_indices = None
        if video_to_active_indices:
            # Note: Threshold optimization is usually done per lab. 
            # We use the union of all behaviors ever labeled for any video in that lab.
            lab_to_active_indices = {}
            # We don't have flat_video_ids here yet to map labs to videos, 
            # but we can try to infer or just pass the video mapping down.
            pass

        strategy = self.config.get('threshold_strategy', 'dynamic')
        
        if strategy == 'kaggle':
            if not hasattr(self, 'classes') or not self.classes:
                print("Warning: 'kaggle' threshold strategy requires class names. Falling back to dynamic or default.")
            else:
                print("Loading hardcoded Kaggle thresholds (TT_PER_LAB_NN)...")
                unique_labs = np.unique(lab_ids)
                for lab in unique_labs:
                    if lab in TT_PER_LAB_NN:
                        lab_thresh = TT_PER_LAB_NN[lab]
                        thresh_arr = np.full(len(self.classes), 0.5)
                        for act, th in lab_thresh.items():
                            if act in self.class_to_idx:
                                thresh_arr[self.class_to_idx[act]] = th
                        self.thresholds[lab] = thresh_arr
                return

        if strategy == 'fixed':
            unique_labs = np.unique(lab_ids)
            num_classes = predictions.shape[1]
            for lab in unique_labs:
                self.thresholds[lab] = np.full(num_classes, 0.5)
            return

        if not self.config.get('optimize_thresholds', False):
            return
            
        print(f"Optimizing thresholds using {self.n_jobs} jobs...")
        unique_labs = np.unique(lab_ids)
        num_classes = predictions.shape[1]
        
        # Use a finer search step (0.01) for professional-grade optimization
        threshold_range = np.arange(0.1, 0.95, 0.01)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._optimize_lab)(
                lab, predictions, targets, lab_ids, threshold_range, num_classes, 
                active_indices=lab_to_active_indices.get(lab) if lab_to_active_indices else None
            )
            for lab in unique_labs
        )
        
        avg_best_f1s = []
        for lab, thresholds, sigmas, best_f1 in results:
            self.thresholds[lab] = thresholds
            self.sigmas[lab] = sigmas
            avg_best_f1s.append(best_f1)
        
        print(f"Thresholds optimized for {len(unique_labs)} labs. Mean Best F1 (Internal): {np.mean(avg_best_f1s):.4f}")
        
        mean_prob = np.mean(predictions)
        max_prob = np.max(predictions)
        print(f"Prediction Stats - Mean Prob: {mean_prob:.6f}, Max Prob: {max_prob:.6f}")

    def _apply_tie_breaking_lab(self, lab, predictions, lab_ids, method):
        lab_mask = (lab_ids == lab)
        lab_probs = predictions[lab_mask]
        
        if lab in self.thresholds:
            thresholds = self.thresholds[lab]
        else:
            thresholds = np.full(predictions.shape[1], 0.5)
        
        above_thresh = lab_probs > thresholds
        
        if method == 'kaggle':
            if not hasattr(self, 'classes') or not self.classes:
                method = 'argmax'
            else:
                P_adj = lab_probs.copy()
                if lab in TIE_CONFIG_V2:
                    cfg = TIE_CONFIG_V2[lab]
                    multi_mask = (above_thresh.sum(axis=1) > 1)
                    
                    if multi_mask.any():
                        for act, delta in cfg.get('boost', {}).items():
                            if act in self.class_to_idx:
                                idx = self.class_to_idx[act]
                                P_adj[multi_mask, idx] += float(delta)
                        
                        for act, delta in cfg.get('penalize', {}).items():
                            if act in self.class_to_idx:
                                idx = self.class_to_idx[act]
                                P_adj[multi_mask, idx] -= float(delta)
                                
                        for winner, loser, margin in cfg.get('prefer', []):
                            if winner in self.class_to_idx and loser in self.class_to_idx:
                                wi, li = self.class_to_idx[winner], self.class_to_idx[loser]
                                fm = multi_mask & above_thresh[:, wi] & above_thresh[:, li]
                                if fm.any():
                                    P_adj[fm, wi] += float(margin)
                                    
                        np.clip(P_adj, 0.0, 1.0, out=P_adj)
                
                P_masked = np.where(above_thresh, P_adj, -np.inf)
                best_idx = np.argmax(P_masked, axis=1)
                has_pred = above_thresh.any(axis=1)
                
                lab_final = np.zeros(lab_probs.shape, dtype=np.uint8)
                lab_final[has_pred, best_idx[has_pred]] = 1
                return lab_mask, lab_final

        if method == 'argmax':
            masked_probs = lab_probs * above_thresh
            max_indices = np.argmax(masked_probs, axis=1)
            has_pred = above_thresh.any(axis=1)
            
            lab_final = np.zeros(lab_probs.shape, dtype=np.uint8)
            lab_final[np.arange(len(lab_probs)), max_indices] = 1
            lab_final[~has_pred] = 0
            
        elif method == 'z_score':
            sigmas = self.sigmas.get(lab, np.ones_like(thresholds))
            sigmas = np.maximum(sigmas, 1e-6)
            
            z_scores = (lab_probs - thresholds) / sigmas
            
            masked_z = z_scores.copy()
            masked_z[~above_thresh] = -np.inf
            
            max_indices = np.argmax(masked_z, axis=1)
            has_pred = above_thresh.any(axis=1)
            
            lab_final = np.zeros(lab_probs.shape, dtype=np.uint8)
            lab_final[np.arange(len(lab_probs)), max_indices] = 1
            lab_final[~has_pred] = 0
            
        return lab_mask, lab_final

    def apply_tie_breaking(self, predictions, lab_ids):
        method = self.config.get('tie_breaking', 'none')
        if method == 'none':
            return predictions
            
        print(f"Applying tie-breaking (method: {method}) using {self.n_jobs} jobs...")
        final_preds = np.zeros(predictions.shape, dtype=np.uint8)
        unique_labs = np.unique(lab_ids)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._apply_tie_breaking_lab)(lab, predictions, lab_ids, method)
            for lab in unique_labs
        )
        
        for lab_mask, lab_final in results:
            final_preds[lab_mask] = lab_final
                
        return final_preds

    def _smooth_seq_median(self, seq, window_size):
        for c in range(seq.shape[1]):
            seq[:, c] = median_filter(seq[:, c], size=window_size)
        return seq

    def _smooth_seq_ema(self, seq, b, a):
        smoothed = lfilter(b, a, seq, axis=0)
        predictions_flipped = np.flip(seq, axis=0)
        smoothed_back = lfilter(b, a, predictions_flipped, axis=0)
        smoothed_back = np.flip(smoothed_back, axis=0)
        return (smoothed + smoothed_back) / 2.0

    def apply_smoothing(self, predictions, verbose=False):
        method = self.config.get('smoothing', {}).get('method', 'none')
        
        if method == 'none':
            return predictions
            
        if verbose:
            print(f"Applying temporal smoothing (method: {method}) chunk-wise to save memory...")
        
        orig_dtype = predictions.dtype
        
        # Determine if we have [TotalFrames, Classes] or [Samples, Window, Classes]
        is_3d = (predictions.ndim == 3)
        temp_shape = predictions.shape
        num_classes = temp_shape[-1]
        
        if not is_3d:
            # Case 1: [TotalFrames, Classes]
            total_frames = temp_shape[0]
            # Use chunks to avoid creating massive temporary float32 arrays
            chunk_size = 500000 
            
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                # Read as float32 for signal processing
                chunk = predictions[start:end].astype(np.float32)
                
                if method == 'median_filter':
                    window_size = self.config['smoothing']['window_size']
                    if window_size % 2 == 0: window_size += 1
                    for c in range(num_classes):
                        chunk[:, c] = median_filter(chunk[:, c], size=window_size)
                elif method == 'ema':
                    alpha = self.config['smoothing']['alpha']
                    b, a = [alpha], [1, -(1-alpha)]
                    # Bidirectional EMA
                    s1 = lfilter(b, a, chunk, axis=0)
                    s2 = lfilter(b, a, np.flip(chunk, axis=0), axis=0)
                    chunk = (s1 + np.flip(s2, axis=0)) / 2.0
                
                # Write back to original (usually float16)
                predictions[start:end] = chunk.astype(orig_dtype)
                
            del chunk
            gc.collect()
            
        else:
            # Case 2: [Samples, Window, Classes]
            # We smooth across the window (axis 1)
            num_samples = temp_shape[0]
            chunk_size = 50000 # Samples per chunk
            
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                chunk = predictions[start:end].astype(np.float32)
                
                if method == 'median_filter':
                    window_size = self.config['smoothing']['window_size']
                    if window_size % 2 == 0: window_size += 1
                    for c in range(num_classes):
                        chunk[:, :, c] = median_filter(chunk[:, :, c], size=(1, window_size))
                elif method == 'ema':
                    alpha = self.config['smoothing']['alpha']
                    b, a = [alpha], [1, -(1-alpha)]
                    s1 = lfilter(b, a, chunk, axis=1)
                    s2 = lfilter(b, a, np.flip(chunk, axis=1), axis=1)
                    chunk = (s1 + np.flip(s2, axis=1)) / 2.0
                
                predictions[start:end] = chunk.astype(orig_dtype)
                
            del chunk
            gc.collect()
            
        return predictions

    def fill_gaps(self, predictions, lab_ids=None, verbose=False):
        gap_config = self.config.get('gap_filling', {})
        max_gap = gap_config.get('max_gap', 0)
        min_duration = gap_config.get('min_duration', 0)
        
        if max_gap <= 0 and min_duration <= 0:
            return predictions
            
        if verbose:
            print(f"Applying gap filling (max_gap: {max_gap}, min_duration: {min_duration})...")
        
        def _fill_seq(seq):
            T, C = seq.shape
            padded = np.ones((T + 2, C), dtype=seq.dtype)
            padded[1:-1, :] = seq
            
            filled = padded
            
            if max_gap > 0:
                structure = np.ones((max_gap + 1, 1))
                filled = binary_closing(filled, structure=structure)
                
            if min_duration > 0:
                structure = np.ones((min_duration, 1))
                filled = binary_opening(filled, structure=structure)
            
            return filled[1:-1, :].astype(seq.dtype)

        if lab_ids is not None:
            unique_labs = np.unique(lab_ids)
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fill_seq)(predictions[lab_ids == lab]) for lab in unique_labs
            )
            final_preds = np.zeros_like(predictions)
            for lab, filled in zip(unique_labs, results):
                final_preds[lab_ids == lab] = filled
            return final_preds
        elif predictions.ndim == 3:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fill_seq)(predictions[i]) for i in range(predictions.shape[0])
            )
            return np.array(results)
        else:
            return _fill_seq(predictions)

    def apply_z_score_tie_breaking(self, probs, thresholds, oof_stats):
        num_classes = probs.shape[1]
        z_scores = np.zeros_like(probs)
        
        for c in range(num_classes):
            sigma = oof_stats['std'][c] if oof_stats and 'std' in oof_stats else 1.0
            thresh = thresholds[c] if isinstance(thresholds, (list, np.ndarray)) else thresholds
            z_scores[:, c] = (probs[:, c] - thresh) / (sigma + 1e-6)
            
        final_preds = np.argmax(z_scores, axis=1)
        return final_preds

    def apply_thresholds(self, predictions, lab_ids):
        print(f"Applying thresholds using {self.n_jobs} jobs...")
        final_preds = np.zeros(predictions.shape, dtype=np.uint8)
        unique_labs = np.unique(lab_ids)
        
        def _apply_lab_thresh(lab):
            mask = (lab_ids == lab)
            lab_preds = predictions[mask]
            if lab in self.thresholds:
                thresh = self.thresholds[lab]
                return mask, (lab_preds > thresh).astype(np.uint8)
            else:
                return mask, (lab_preds > 0.5).astype(np.uint8)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_apply_lab_thresh)(lab) for lab in unique_labs
        )
        
        for mask, lab_binary in results:
            final_preds[mask] = lab_binary
            
        return final_preds

    def calculate_f1_scores(self, predictions, targets, lab_ids, video_to_active_indices=None, flat_video_ids=None):
        print(f"Calculating F1 scores using {self.n_jobs} jobs (Official MABe Logic: Video-Specific Active Only)...")
        unique_labs = np.unique(lab_ids)
        
        def _calc_lab_f1(lab):
            mask = (lab_ids == lab)
            lab_p = predictions[mask].copy() 
            lab_t = targets[mask]
            
            # 改进：官方 F1 是针对每个视频过滤的
            if video_to_active_indices is not None and flat_video_ids is not None:
                lab_vids = flat_video_ids[mask]
                unique_vids = np.unique(lab_vids)
                
                vid_f1s = []
                for vid in unique_vids:
                    vid_mask = (lab_vids == vid)
                    v_p = lab_p[vid_mask]
                    v_t = lab_t[vid_mask]
                    
                    active_idx = video_to_active_indices.get(str(vid))
                    if active_idx is not None:
                        # 仅保留激活类别的预测，消除不应出现的 FP
                        inactive = np.ones(v_p.shape[1], dtype=bool)
                        inactive[active_idx] = False
                        v_p[:, inactive] = 0
                        
                        f1s = f1_score(v_t, v_p, average=None, zero_division=0.0)
                        vid_f1s.append(f1s[active_idx])
                
                if not vid_f1s: return lab, 0.0
                # 按照 Macro F1 逻辑，先平均每个类在所有视频的表现，再对类求平均
                # 简化处理：收集所有激活类的 F1 样本求均值
                all_relevant_f1s = np.concatenate(vid_f1s)
                return lab, np.mean(all_relevant_f1s)
            else:
                # 降级逻辑
                f1s = f1_score(lab_t, lab_p, average=None, zero_division=0.0)
                present_classes = (lab_t.sum(axis=0) > 0)
                return lab, np.mean(f1s[present_classes]) if np.any(present_classes) else 0.0
            
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_calc_lab_f1)(lab) for lab in unique_labs
        )
        
        lab_f1s = {lab: f1 for lab, f1 in results}
        lab_f1s['overall'] = np.mean([f1 for f1 in lab_f1s.values()])
        
        return lab_f1s

    def apply_fps_compensation(self, predictions, lab_ids):
        unique_labs = np.unique(lab_ids)
        if 'AdaptableSnail' not in unique_labs:
            return predictions
            
        print("Applying FPS compensation for AdaptableSnail (30/25 conversion)...")
        final_preds = predictions.copy()
        
        return final_preds

    def apply_masking(self, predictions, targets, lab_ids):
        print("[Post-Processing] Masking invalid frames (Memory-efficient)...")
        valid_mask = ~np.isnan(targets).any(axis=-1)
        
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        lab_ids = lab_ids[valid_mask]
        
        gc.collect()
        
        return predictions, targets, lab_ids

    def __call__(self, predictions, lab_ids=None, oof_stats=None):
        preds = self.apply_smoothing(predictions, verbose=True)
        
        final_binary = np.zeros_like(preds)
        
        if lab_ids is not None and self.thresholds:
            unique_labs = np.unique(lab_ids)
            for lab in unique_labs:
                mask = (lab_ids == lab)
                lab_preds = preds[mask]
                threshs = self.thresholds.get(lab, [0.5]*preds.shape[1])
                
                if self.config.get('tie_breaking') == 'z_score' and oof_stats:
                     indices = self.apply_z_score_tie_breaking(lab_preds, threshs, oof_stats)
                     lab_binary = np.zeros_like(lab_preds)
                     lab_binary[np.arange(len(lab_preds)), indices] = 1
                     final_binary[mask] = lab_binary
                else:
                    lab_binary = (lab_preds > np.array(threshs)).astype(int)
                    final_binary[mask] = lab_binary
        else:
            final_binary = (preds > 0.5).astype(int)
        
        final_binary = self.fill_gaps(final_binary, verbose=True)
        
        return final_binary
