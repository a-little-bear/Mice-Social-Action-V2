import torch
import numpy as np

class FeatureGenerator:
    def __init__(self, config):
        self.config = config
        self.use_velocity = config.get('use_velocity', True)
        self.use_acceleration = config.get('use_acceleration', True)
        self.use_distances = config.get('use_distances', True)
        self.use_jerk = config.get('use_jerk', False)
        self.use_angles = config.get('use_angles', False)
        self.use_window_stats = config.get('use_window_stats', False)
        self.window_sizes = config.get('window_sizes', [5, 15, 30])

    def compute_distances(self, keypoints):
        if keypoints.dim() == 4:
            T, M, K, C = keypoints.shape
            flat_kps = keypoints.view(T, M*K, C)
        else:
            flat_kps = keypoints

        diff = flat_kps.unsqueeze(2) - flat_kps.unsqueeze(1)
        dist_matrix = torch.norm(diff, dim=-1)
        
        num_points = dist_matrix.shape[1]
        triu_indices = torch.triu_indices(num_points, num_points, offset=1)
        distances = dist_matrix[:, triu_indices[0], triu_indices[1]]
        
        return distances

    def compute_derivatives(self, keypoints):
        velocity = torch.zeros_like(keypoints)
        velocity[1:] = keypoints[1:] - keypoints[:-1]
        
        acceleration = torch.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        jerk = torch.zeros_like(acceleration)
        jerk[1:] = acceleration[1:] - acceleration[:-1]
        
        return velocity, acceleration, jerk

    def compute_angles(self, keypoints):
        if keypoints.ndim == 4:
            nose = keypoints[:, :, 0, :]
            tail = keypoints[:, :, 3, :]
        else:
            nose = keypoints[:, 0, :]
            tail = keypoints[:, 3, :]
            
        vec = nose - tail
        angles = torch.atan2(vec[..., 1], vec[..., 0])
        
        return angles.unsqueeze(-1)

    def compute_window_stats(self, features):
        stats_list = []
        features_t = features.transpose(0, 1).unsqueeze(0)
        
        for w in self.window_sizes:
            padding = w // 2
            avg = torch.nn.functional.avg_pool1d(features_t, kernel_size=w, stride=1, padding=padding)
            max_val = torch.nn.functional.max_pool1d(features_t, kernel_size=w, stride=1, padding=padding)
            
            if avg.shape[2] != features.shape[0]:
                avg = avg[:, :, :features.shape[0]]
                max_val = max_val[:, :, :features.shape[0]]
                
            stats_list.append(avg.squeeze(0).transpose(0, 1))
            stats_list.append(max_val.squeeze(0).transpose(0, 1))
            
        return torch.cat(stats_list, dim=-1)

    def get_feature_dim(self, num_mice, num_keypoints):
        dim = num_mice * num_keypoints * 2
        
        if not self.config.get('enable_strong_features', True):
            return dim

        if self.use_distances:
            num_points = num_mice * num_keypoints
            dim += (num_points * (num_points - 1)) // 2
            
        if self.use_velocity:
            dim += num_mice * num_keypoints * 2
        if self.use_acceleration:
            dim += num_mice * num_keypoints * 2
        if self.use_jerk:
            dim += num_mice * num_keypoints * 2
        if self.use_angles:
            dim += num_mice
            
        if self.use_window_stats:
            base_dim = dim
            num_stats = 2 * len(self.window_sizes)
            dim += base_dim * num_stats
            
        return dim

    def __call__(self, keypoints):
        T, M, K, C = keypoints.shape
        
        if not self.config.get('enable_strong_features', True):
            return keypoints.view(T, -1)
            
        feature_list = [keypoints.view(T, -1)]
        
        if self.use_distances:
            feature_list.append(self.compute_distances(keypoints))
            
        velocity, acceleration, jerk = self.compute_derivatives(keypoints)
        
        if self.use_velocity:
            feature_list.append(velocity.view(T, -1))
        if self.use_acceleration:
            feature_list.append(acceleration.view(T, -1))
        if self.use_jerk:
            feature_list.append(jerk.view(T, -1))
            
        if self.use_angles:
            feature_list.append(self.compute_angles(keypoints).view(T, -1))
            
        combined_features = torch.cat(feature_list, dim=-1)
        
        if self.use_window_stats:
            window_stats = self.compute_window_stats(combined_features)
            combined_features = torch.cat([combined_features, window_stats], dim=-1)
            
        return combined_features
