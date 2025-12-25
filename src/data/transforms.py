import numpy as np
from scipy.ndimage import zoom

class CoordinateTransform:
    def __init__(self, view='egocentric', nose_idx=0, tail_base_idx=3):
        self.view = view
        self.nose_idx = nose_idx
        self.tail_base_idx = tail_base_idx

    def __call__(self, keypoints):
        if self.view == 'raw':
            return keypoints
            
        if keypoints.ndim == 4:
            agent_kps = keypoints[:, 0, :, :]
        else:
            agent_kps = keypoints
            
        center = agent_kps[:, self.tail_base_idx, :]
        
        if keypoints.ndim == 4:
            center_expanded = center[:, None, None, :]
        else:
            center_expanded = center[:, None, :]
            
        centered_kps = keypoints - center_expanded
        
        if self.view == 'centered':
            return centered_kps
            
        if self.view == 'egocentric':
            spine_vec = agent_kps[:, self.nose_idx, :] - agent_kps[:, self.tail_base_idx, :]
            theta = np.arctan2(spine_vec[:, 1], spine_vec[:, 0])
            rotation_angle = np.pi/2 - theta
            
            c = np.cos(rotation_angle)
            s = np.sin(rotation_angle)
            
            R = np.stack([np.stack([c, -s], axis=-1), 
                          np.stack([s, c], axis=-1)], axis=-2)
            
            T_dim = keypoints.shape[0]
            original_shape = keypoints.shape
            
            flat_kps = centered_kps.reshape(T_dim, -1, 2)
            rotated_kps = np.einsum('tij,tnj->tni', R, flat_kps)
            
            return rotated_kps.reshape(original_shape)
            
        return keypoints

class BodyPartMapping:
    def __init__(self, enabled=True):
        self.enabled = enabled
        
        # Target 7-point skeleton
        self.target_parts = ['nose', 'ear_left', 'ear_right', 'neck', 'side_left', 'side_right', 'tail_base']
        
        # Unification Strategy
        self.unification_map = {
            'nose': ['nose', 'head'],
            'ear_left': ['ear_left'],
            'ear_right': ['ear_right'],
            'neck': ['neck', 'body_center', 'spine_1'],
            'side_left': ['hip_left', 'lateral_left', 'body_center', 'neck'], # Fallback to center
            'side_right': ['hip_right', 'lateral_right', 'body_center', 'neck'], # Fallback to center
            'tail_base': ['tail_base']
        }
        
        # Lab-specific body parts (Derived from EDA)
        self.lab_configs = {
            'AdaptableSnail': ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip'],
            'BoisterousParrot': ['body_center', 'ear_left', 'ear_right', 'nose', 'tail_base'],
            'CalMS21_supplemental': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'CalMS21_task1': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'CalMS21_task2': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'CautiousGiraffe': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'CRIM13': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'DeliriousFly': ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'nose', 'tail_base', 'tail_tip'],
            'ElegantMink': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'GroovyShrew': ['ear_left', 'ear_right', 'head', 'tail_base'],
            'InvincibleJellyfish': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'JovialSwallow': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'LyricalHare': ['ear_left', 'ear_right', 'nose', 'tail_base', 'tail_tip'],
            'MABe22_keypoints': ['body_center', 'ear_left', 'ear_right', 'forepaw_left', 'forepaw_right', 'hindpaw_left', 'hindpaw_right', 'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip'],
            'MABe22_movies': ['body_center', 'ear_left', 'ear_right', 'forepaw_left', 'forepaw_right', 'hindpaw_left', 'hindpaw_right', 'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip'],
            'NiftyGoldfinch': ['body_center', 'ear_left', 'ear_right', 'nose', 'tail_base'],
            'PleasantMeerkat': ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'nose', 'tail_base', 'tail_tip'],
            'ReflectiveManatee': ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'nose', 'tail_base'],
            'SparklingTapir': ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'nose', 'tail_base'],
            'TranquilPanther': ['ear_left', 'ear_right', 'hip_left', 'hip_right', 'neck', 'nose', 'tail_base'],
            'UppityFerret': ['body_center', 'ear_left', 'ear_right', 'hip_left', 'hip_right', 'lateral_left', 'lateral_right', 'nose', 'spine_1', 'spine_2', 'tail_base', 'tail_middle_1', 'tail_middle_2', 'tail_tip']
        }

    def __call__(self, keypoints, lab_id):
        """
        keypoints: (T, num_mice, num_parts, 2) or (T, num_parts, 2)
        lab_id: str
        """
        if not self.enabled:
            return keypoints
            
        if lab_id not in self.lab_configs:
            # Fallback for unknown labs: return as is or try to guess?
            # For now, return as is to avoid crashing, but warn
            return keypoints

        source_parts = self.lab_configs[lab_id]
        
        # Handle input shape
        if keypoints.ndim == 4:
            # (T, M, P, 2)
            T, M, P, C = keypoints.shape
            new_kps = np.zeros((T, M, len(self.target_parts), C))
        else:
            # (T, P, 2)
            T, P, C = keypoints.shape
            M = 1
            new_kps = np.zeros((T, len(self.target_parts), C))
            keypoints = keypoints[:, None, :, :] # Add mouse dim temporarily

        # Construct the new skeleton
        for i, target in enumerate(self.target_parts):
            # 1. Check for Virtual Neck (GroovyShrew, LyricalHare)
            if target == 'neck' and lab_id in ['GroovyShrew', 'LyricalHare']:
                # Compute mean of ears
                try:
                    idx_l = source_parts.index('ear_left')
                    idx_r = source_parts.index('ear_right')
                    new_kps[:, :, i, :] = (keypoints[:, :, idx_l, :] + keypoints[:, :, idx_r, :]) / 2
                    continue
                except ValueError:
                    pass # Fallback to standard mapping if ears missing (unlikely)

            # 2. Standard Mapping with Fallback
            candidates = self.unification_map[target]
            found = False
            for cand in candidates:
                if cand in source_parts:
                    idx = source_parts.index(cand)
                    new_kps[:, :, i, :] = keypoints[:, :, idx, :]
                    found = True
                    break
            
            if not found:
                # If still not found (should not happen with our fallbacks), fill with zeros or NaN
                pass

        if M == 1 and new_kps.shape[1] == 1:
            return new_kps[:, 0, :, :]
            
        return new_kps



class FPSCorrection:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps

    def __call__(self, keypoints, lab_id, current_fps=None):
        # If current_fps is not provided, fallback to lab-based defaults
        if current_fps is None:
            if lab_id == 'AdaptableSnail':
                current_fps = 25
            else:
                current_fps = 30
        
        if abs(current_fps - self.target_fps) < 0.1:
            return keypoints
            
        T = keypoints.shape[0]
        target_T = int(T * (self.target_fps / current_fps))
        
        if T == target_T:
            return keypoints
            
        # Vectorized interpolation using scipy.ndimage.zoom
        # keypoints: (T, M, P, 2)
        zoom_factors = (target_T / T, 1, 1, 1)
        return zoom(keypoints, zoom_factors, order=1)

class Augmentation:
    def __init__(self, config):
        self.config = config

    def flip(self, keypoints):
        keypoints[..., 0] = -keypoints[..., 0]
        if keypoints.shape[-2] > 2:
            temp = keypoints[..., 1, :].copy()
            keypoints[..., 1, :] = keypoints[..., 2, :]
            keypoints[..., 2, :] = temp
        return keypoints

    def __call__(self, keypoints):
        if not self.config.get('enabled', False):
            return keypoints

        if self.config.get('flip') and np.random.rand() > 0.5:
            keypoints = self.flip(keypoints)
            
        if self.config.get('rotate') and np.random.rand() > 0.5:
            theta = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            shape = keypoints.shape
            flat = keypoints.reshape(-1, 2)
            flat = flat @ R.T
            keypoints = flat.reshape(shape)
            
        if self.config.get('time_stretch') and np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            T = keypoints.shape[0]
            
            # Vectorized time stretch using zoom
            zoom_factors = (factor, 1, 1, 1)
            stretched = zoom(keypoints, zoom_factors, order=1)
            
            # Resample back to original T to maintain window size
            new_T = stretched.shape[0]
            resample_factors = (T / new_T, 1, 1, 1)
            keypoints = zoom(stretched, resample_factors, order=1)
            
            # Ensure exact T due to rounding
            if keypoints.shape[0] > T:
                keypoints = keypoints[:T]
            elif keypoints.shape[0] < T:
                pad = np.zeros((T - keypoints.shape[0],) + keypoints.shape[1:], dtype=keypoints.dtype)
                keypoints = np.concatenate([keypoints, pad], axis=0)

        if self.config.get('noise', 0.0) > 0:
            noise_level = self.config['noise']
            noise = np.random.normal(0, noise_level, keypoints.shape)
            keypoints += noise
            
        return keypoints


