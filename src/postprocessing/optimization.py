import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import f1_score

class PostProcessor:
    """
    Module IV: Post-Processing Refiner (PPR)
    """
    def __init__(self, config):
        self.config = config
        self.thresholds = {} # Dictionary to store lab-specific thresholds
        self.sigmas = {} # Dictionary to store lab-specific prediction std devs (for Z-score)

    def optimize_thresholds(self, predictions, targets, lab_ids):
        """
        Search for optimal thresholds per lab using Grid Search.
        predictions: [N, C] probabilities
        targets: [N, C] binary labels
        lab_ids: [N]
        """
        if not self.config.get('optimize_thresholds', False):
            return
            
        print("Optimizing thresholds...")
        unique_labs = np.unique(lab_ids)
        num_classes = predictions.shape[1]
        
        # Grid search range
        threshold_range = np.arange(0.1, 0.95, 0.05)
        
        for lab in unique_labs:
            lab_mask = (lab_ids == lab)
            lab_preds = predictions[lab_mask]
            lab_targets = targets[lab_mask]
            
            best_thresh_per_class = []
            sigmas_per_class = []
            
            for c in range(num_classes):
                # Calculate sigma for Z-score
                sigmas_per_class.append(np.std(lab_preds[:, c]))

                best_f1 = -1
                best_th = 0.5
                
                # If no positive samples, skip or set default
                if np.sum(lab_targets[:, c]) == 0:
                    best_thresh_per_class.append(0.5)
                    continue
                
                for th in threshold_range:
                    binary_preds = (lab_preds[:, c] > th).astype(int)
                    f1 = f1_score(lab_targets[:, c], binary_preds)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                
                best_thresh_per_class.append(best_th)
            
            self.thresholds[lab] = np.array(best_thresh_per_class)
            self.sigmas[lab] = np.array(sigmas_per_class)
            print(f"Lab {lab} thresholds: {best_thresh_per_class}")

    def apply_tie_breaking(self, predictions, lab_ids):
        """
        Apply tie-breaking logic when multiple classes are above threshold.
        predictions: [N, C] probabilities
        lab_ids: [N]
        """
        method = self.config.get('tie_breaking', 'none')
        if method == 'none':
            return predictions
            
        final_preds = np.zeros_like(predictions)
        unique_labs = np.unique(lab_ids)
        
        for lab in unique_labs:
            if lab not in self.thresholds:
                continue
                
            lab_mask = (lab_ids == lab)
            lab_probs = predictions[lab_mask]
            thresholds = self.thresholds[lab]
            
            # Binary mask of classes above threshold
            above_thresh = lab_probs > thresholds
            
            if method == 'argmax':
                # Just take the max prob among those above threshold
                # If none above threshold, all 0
                # If multiple, max prob wins
                # We can implement this by zeroing out below threshold and taking argmax
                masked_probs = lab_probs * above_thresh
                max_indices = np.argmax(masked_probs, axis=1)
                # Check if row has any valid prediction
                has_pred = above_thresh.any(axis=1)
                
                # Create one-hot
                lab_final = np.zeros_like(lab_probs)
                lab_final[np.arange(len(lab_probs)), max_indices] = 1
                lab_final[~has_pred] = 0
                
                final_preds[lab_mask] = lab_final
                
            elif method == 'z_score':
                sigmas = self.sigmas.get(lab, np.ones_like(thresholds))
                # Avoid div by zero
                sigmas = np.maximum(sigmas, 1e-6)
                
                z_scores = (lab_probs - thresholds) / sigmas
                
                # Only consider those above threshold (z_score > 0)
                masked_z = z_scores * above_thresh
                masked_z[~above_thresh] = -np.inf
                
                max_indices = np.argmax(masked_z, axis=1)
                has_pred = above_thresh.any(axis=1)
                
                lab_final = np.zeros_like(lab_probs)
                lab_final[np.arange(len(lab_probs)), max_indices] = 1
                lab_final[~has_pred] = 0
                
                final_preds[lab_mask] = lab_final
                
        return final_preds

    def apply_smoothing(self, predictions):
        """
        Apply temporal smoothing (EMA or Median).
        predictions: [T, C]
        """
        method = self.config.get('smoothing', {}).get('method', 'none')
        
        if method == 'none':
            return predictions
            
        if method == 'median_filter':
            window_size = self.config['smoothing']['window_size']
            # Apply median filter along time dimension for each class
            smoothed = np.zeros_like(predictions)
            for c in range(predictions.shape[1]):
                smoothed[:, c] = medfilt(predictions[:, c], kernel_size=window_size)
            return smoothed
            
        elif method == 'ema':
            alpha = self.config['smoothing']['alpha']
            # Simple EMA implementation
            # y[t] = alpha * x[t] + (1-alpha) * y[t-1]
            # Can use pandas ewm for speed if available, or loop
            # Forward pass
            smoothed = np.zeros_like(predictions)
            smoothed[0] = predictions[0]
            for t in range(1, len(predictions)):
                smoothed[t] = alpha * predictions[t] + (1-alpha) * smoothed[t-1]
            
            # Backward pass (Bidirectional smoothing)
            smoothed_back = np.zeros_like(predictions)
            smoothed_back[-1] = predictions[-1]
            for t in range(len(predictions)-2, -1, -1):
                smoothed_back[t] = alpha * predictions[t] + (1-alpha) * smoothed_back[t+1]
                
            return (smoothed + smoothed_back) / 2.0
        
        return predictions

    def fill_gaps(self, predictions):
        """
        Fill short gaps in binary predictions.
        predictions: [T, C] binary
        """
        max_gap = self.config.get('gap_filling', {}).get('max_gap', 0)
        if max_gap <= 0:
            return predictions
            
        filled_preds = predictions.copy()
        T, C = predictions.shape
        
        for c in range(C):
            # Find runs of 0s
            binary = predictions[:, c]
            padded = np.concatenate(([1], binary, [1]))
            diff = np.diff(padded)
            starts = np.where(diff == -1)[0]
            ends = np.where(diff == 1)[0]
            
            for s, e in zip(starts, ends):
                length = e - s
                if length <= max_gap:
                    filled_preds[s:e, c] = 1
                    
        return filled_preds

        min_duration = self.config['gap_filling']['min_duration']
        
        # Simple morphological closing-like operation or iteration
        # For simplicity, iterating
        T, C = predictions.shape
        filled = predictions.copy()
        
        for c in range(C):
            # Fill gaps
            # Find 0s between 1s with length <= max_gap
            # This is a simplified logic. 
            # In production, use scipy.ndimage.binary_closing
            pass # Placeholder for complex logic, keeping simple for now
            
        return filled

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

    def __call__(self, predictions, lab_ids=None, oof_stats=None):
        # predictions: [N, C] probabilities
        
        # 1. Smoothing (on probabilities)
        preds = self.apply_smoothing(predictions)
        
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
        final_binary = self.fill_gaps(final_binary)
        
        return final_binary
