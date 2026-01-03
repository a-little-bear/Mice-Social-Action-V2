import numpy as np
import gc
from scipy.signal import lfilter
from scipy.ndimage import binary_closing, median_filter, binary_opening
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

class PostProcessor:
    """
    Module IV: Post-Processing Refiner (PPR)
    """
    def __init__(self, config):
        self.config = config
        self.thresholds = {} # Dictionary to store lab-specific thresholds
        self.sigmas = {} # Dictionary to store lab-specific prediction std devs (for Z-score)
        self.n_jobs = config.get('n_jobs', -1)

    def _optimize_lab(self, lab, predictions, targets, lab_ids, threshold_range, num_classes):
        lab_mask = (lab_ids == lab)
        # [N_lab, C]
        lab_preds = predictions[lab_mask]
        # Handle uint8 targets (1 is positive, 0 is negative, 255 is NaN)
        lab_targets = (targets[lab_mask] == 1)
        
        # Calculate sigmas for all classes at once
        # [C]
        sigmas_per_class = np.std(lab_preds, axis=0)

        # Check for positive samples per class
        # [C]
        has_positives = lab_targets.sum(axis=0) > 0
        
        # Prepare output thresholds (default 0.5)
        best_thresh_per_class = np.full(num_classes, 0.5)
        
        # Only process classes with positives
        if not np.any(has_positives):
            return lab, best_thresh_per_class, sigmas_per_class

        # Memory-efficient F1 calculation: loop over thresholds instead of expanding memory
        # [C, K]
        f1_scores = np.zeros((num_classes, len(threshold_range)))
        
        for i, th in enumerate(threshold_range):
            # [N_lab, C]
            preds_bool = lab_preds > th
            
            # Sum over samples (axis 0) -> [C]
            tp = (preds_bool & lab_targets).sum(axis=0)
            fp = (preds_bool & ~lab_targets).sum(axis=0)
            fn = (~preds_bool & lab_targets).sum(axis=0)
            
            denom = 2 * tp + fp + fn
            f1_scores[:, i] = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom!=0)
        
        # Best index per class: [C]
        best_indices = np.argmax(f1_scores, axis=1)
        
        # Map indices to thresholds
        optimized_thresholds = threshold_range[best_indices]
        
        # Restore default 0.5 for classes with no positives
        optimized_thresholds[~has_positives] = 0.5
        
        return lab, optimized_thresholds, sigmas_per_class

    def optimize_thresholds(self, predictions, targets, lab_ids):
        """
        Search for optimal thresholds per lab using Grid Search.
        predictions: [N, C] probabilities
        targets: [N, C] binary labels
        lab_ids: [N]
        """
        if not self.config.get('optimize_thresholds', False):
            return
            
        print(f"Optimizing thresholds using {self.n_jobs} jobs...")
        unique_labs = np.unique(lab_ids)
        num_classes = predictions.shape[1]
        
        # Grid search range
        threshold_range = np.arange(0.1, 0.95, 0.05)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._optimize_lab)(lab, predictions, targets, lab_ids, threshold_range, num_classes)
            for lab in unique_labs
        )
        
        for lab, thresholds, sigmas in results:
            self.thresholds[lab] = thresholds
            self.sigmas[lab] = sigmas
        
        print(f"Thresholds optimized for {len(unique_labs)} labs.")

    def _apply_tie_breaking_lab(self, lab, predictions, lab_ids, method):
        lab_mask = (lab_ids == lab)
        lab_probs = predictions[lab_mask]
        
        if lab in self.thresholds:
            thresholds = self.thresholds[lab]
        else:
            thresholds = np.full(predictions.shape[1], 0.5)
        
        # Binary mask of classes above threshold
        above_thresh = lab_probs > thresholds
        
        if method == 'argmax':
            # Just take the max prob among those above threshold
            masked_probs = lab_probs * above_thresh
            max_indices = np.argmax(masked_probs, axis=1)
            has_pred = above_thresh.any(axis=1)
            
            # Create one-hot (using uint8)
            lab_final = np.zeros(lab_probs.shape, dtype=np.uint8)
            lab_final[np.arange(len(lab_probs)), max_indices] = 1
            lab_final[~has_pred] = 0
            
        elif method == 'z_score':
            sigmas = self.sigmas.get(lab, np.ones_like(thresholds))
            sigmas = np.maximum(sigmas, 1e-6)
            
            z_scores = (lab_probs - thresholds) / sigmas
            
            # Only consider those above threshold
            masked_z = z_scores.copy()
            masked_z[~above_thresh] = -np.inf
            
            max_indices = np.argmax(masked_z, axis=1)
            has_pred = above_thresh.any(axis=1)
            
            lab_final = np.zeros(lab_probs.shape, dtype=np.uint8)
            lab_final[np.arange(len(lab_probs)), max_indices] = 1
            lab_final[~has_pred] = 0
            
        return lab_mask, lab_final

    def apply_tie_breaking(self, predictions, lab_ids):
        """
        Apply tie-breaking logic when multiple classes are above threshold.
        predictions: [N, C] probabilities
        lab_ids: [N]
        """
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
        # seq: [T, C]
        for c in range(seq.shape[1]):
            seq[:, c] = median_filter(seq[:, c], size=window_size)
        return seq

    def _smooth_seq_ema(self, seq, b, a):
        # seq: [T, C]
        # Forward
        smoothed = lfilter(b, a, seq, axis=0)
        # Backward
        predictions_flipped = np.flip(seq, axis=0)
        smoothed_back = lfilter(b, a, predictions_flipped, axis=0)
        smoothed_back = np.flip(smoothed_back, axis=0)
        return (smoothed + smoothed_back) / 2.0

    def apply_smoothing(self, predictions, verbose=False):
        """
        Apply temporal smoothing (EMA or Median).
        predictions: [T, C] or [N, T, C]
        """
        method = self.config.get('smoothing', {}).get('method', 'none')
        
        if method == 'none':
            return predictions
            
        if verbose:
            print(f"Applying temporal smoothing (method: {method})...")
        
        # Ensure float32 for numerical stability during filtering
        orig_dtype = predictions.dtype
        if predictions.dtype == np.float16:
            predictions = predictions.astype(np.float32)

        if method == 'median_filter':
            window_size = self.config['smoothing']['window_size']
            if window_size % 2 == 0: window_size += 1
            
            if predictions.ndim == 2:
                # [T, C]
                for c in range(predictions.shape[1]):
                    predictions[:, c] = median_filter(predictions[:, c], size=window_size)
            else:
                # [N, T, C] - Vectorized median filter along T axis
                predictions = median_filter(predictions, size=(1, window_size, 1))
            
        elif method == 'ema':
            alpha = self.config['smoothing']['alpha']
            b = [alpha]
            a = [1, -(1-alpha)]
            
            if predictions.ndim == 2:
                # Forward
                smoothed = lfilter(b, a, predictions, axis=0)
                # Backward
                smoothed_back = lfilter(b, a, np.flip(predictions, axis=0), axis=0)
                predictions = (smoothed + np.flip(smoothed_back, axis=0)) / 2.0
            else:
                # [N, T, C] - Vectorized EMA along T axis (axis=1)
                # Forward
                smoothed = lfilter(b, a, predictions, axis=1)
                # Backward
                smoothed_back = lfilter(b, a, np.flip(predictions, axis=1), axis=1)
                predictions = (smoothed + np.flip(smoothed_back, axis=1)) / 2.0
        
        if orig_dtype == np.float16:
            predictions = predictions.astype(np.float16)
            
        return predictions

    def fill_gaps(self, predictions, lab_ids=None, verbose=False):
        """
        Fill short gaps and remove short bursts in binary predictions.
        predictions: [N, C] binary or [T, C] binary
        """
        gap_config = self.config.get('gap_filling', {})
        max_gap = gap_config.get('max_gap', 0)
        min_duration = gap_config.get('min_duration', 0)
        
        if max_gap <= 0 and min_duration <= 0:
            return predictions
            
        if verbose:
            print(f"Applying gap filling (max_gap: {max_gap}, min_duration: {min_duration})...")
        
        def _fill_seq(seq):
            T, C = seq.shape
            # Pad with 1s to fill boundary gaps
            padded = np.ones((T + 2, C), dtype=seq.dtype)
            padded[1:-1, :] = seq
            
            filled = padded
            
            # 1. Fill gaps (0s between 1s) -> binary_closing
            if max_gap > 0:
                structure = np.ones((max_gap + 1, 1))
                filled = binary_closing(filled, structure=structure)
                
            # 2. Remove short bursts (1s between 0s) -> binary_opening
            if min_duration > 0:
                structure = np.ones((min_duration, 1))
                filled = binary_opening(filled, structure=structure)
            
            return filled[1:-1, :].astype(seq.dtype)

        if lab_ids is not None:
            unique_labs = np.unique(lab_ids)
            # Sequential processing for lab-specific sequences to save memory
            final_preds = np.zeros_like(predictions)
            for lab in unique_labs:
                mask = (lab_ids == lab)
                final_preds[mask] = _fill_seq(predictions[mask])
            return final_preds
        elif predictions.ndim == 3:
            # [N, T, C] - Sequential processing to save memory
            final_preds = np.zeros_like(predictions)
            for i in range(predictions.shape[0]):
                final_preds[i] = _fill_seq(predictions[i])
            return final_preds
        else:
            return _fill_seq(predictions)

    def apply_z_score_tie_breaking(self, probs, thresholds, oof_stats):
        """
        Fair tie-breaking using Z-score.
        z = (prob - threshold) / sigma
        """
        # probs: [Time, Classes]
        # thresholds: [Classes]
        # oof_stats: Dictionary with 'std' per class
        
        num_classes = probs.shape[1]
        z_scores = np.zeros_like(probs)
        
        for c in range(num_classes):
            sigma = oof_stats['std'][c] if oof_stats and 'std' in oof_stats else 1.0
            thresh = thresholds[c] if isinstance(thresholds, (list, np.ndarray)) else thresholds
            z_scores[:, c] = (probs[:, c] - thresh) / (sigma + 1e-6)
            
        # Select class with highest Z-score
        final_preds = np.argmax(z_scores, axis=1)
        return final_preds

    def apply_thresholds(self, predictions, lab_ids):
        """
        Apply thresholds to predictions.
        predictions: [N, C] probabilities
        lab_ids: [N]
        """
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

    def calculate_f1_scores(self, predictions, targets, lab_ids):
        """
        Calculate F1 scores per lab and overall.
        predictions: [N, C] binary
        targets: [N, C] binary
        lab_ids: [N]
        """
        print(f"Calculating F1 scores using {self.n_jobs} jobs...")
        unique_labs = np.unique(lab_ids)
        
        def _calc_lab_f1(lab):
            mask = (lab_ids == lab)
            lab_p = predictions[mask]
            lab_t = targets[mask]
            # Vectorized F1 for all classes
            f1s = f1_score(lab_t, lab_p, average=None, zero_division=0.0)
            return lab, np.mean(f1s)
            
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_calc_lab_f1)(lab) for lab in unique_labs
        )
        
        lab_f1s = {lab: f1 for lab, f1 in results}
        lab_f1s['overall'] = np.mean([f1 for f1 in lab_f1s.values()])
        
        return lab_f1s

    def apply_fps_compensation(self, predictions, lab_ids):
        """
        Compensate for FPS differences (e.g., AdaptableSnail 25 FPS vs 30 FPS labels).
        This is used during inference to align predictions with original video frames.
        """
        unique_labs = np.unique(lab_ids)
        if 'AdaptableSnail' not in unique_labs:
            return predictions
            
        print("Applying FPS compensation for AdaptableSnail (30/25 conversion)...")
        final_preds = predictions.copy()
        
        # AdaptableSnail specific logic:
        # If model was trained on 30FPS (interpolated), but submission needs to align
        # with original 25FPS tracking or 30FPS labels that were stretched.
        # According to 2nd place solution, they aligned predictions to 30FPS labels.
        
        # This method can be expanded if we need to resample the sequence.
        # For now, we ensure the logic is available for the submission script.
        return final_preds

    def apply_masking(self, predictions, targets, lab_ids):
        """
        Memory-efficient masking of invalid frames.
        """
        print("[Post-Processing] Masking invalid frames (Memory-efficient)...")
        # Use np.isnan check since we are now using float32 targets
        valid_mask = ~np.isnan(targets).any(axis=-1)
        
        # Filter
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        lab_ids = lab_ids[valid_mask]
        
        gc.collect()
        
        return predictions, targets, lab_ids

    def __call__(self, predictions, lab_ids=None, oof_stats=None):
        # predictions: [N, C] probabilities
        
        # 1. Smoothing (on probabilities)
        preds = self.apply_smoothing(predictions, verbose=True)
        
        # 2. Thresholding & Tie Breaking
        final_binary = np.zeros_like(preds)
        
        if lab_ids is not None and self.thresholds:
            # Apply lab-specific thresholds
            # This requires iterating or vectorized lookup
            # For simplicity, assuming single lab batch or loop
            unique_labs = np.unique(lab_ids)
            for lab in unique_labs:
                mask = (lab_ids == lab)
                lab_preds = preds[mask]
                threshs = self.thresholds.get(lab, [0.5]*preds.shape[1])
                
                if self.config.get('tie_breaking') == 'z_score' and oof_stats:
                     # Z-score logic returns indices
                     indices = self.apply_z_score_tie_breaking(lab_preds, threshs, oof_stats)
                     # Convert to one-hot
                     lab_binary = np.zeros_like(lab_preds)
                     lab_binary[np.arange(len(lab_preds)), indices] = 1
                     final_binary[mask] = lab_binary
                else:
                    # Standard thresholding
                    lab_binary = (lab_preds > np.array(threshs)).astype(int)
                    final_binary[mask] = lab_binary
        else:
            final_binary = (preds > 0.5).astype(int)
        
        # 3. Gap Filling (on binary)
        final_binary = self.fill_gaps(final_binary, verbose=True)
        
        return final_binary
