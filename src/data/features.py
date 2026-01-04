import torch
import torch.nn as nn
import numpy as np

class FeatureGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_velocity = config.get('use_velocity', True)
        self.use_acceleration = config.get('use_acceleration', True)
        self.use_distances = config.get('use_distances', True)
        self.use_jerk = config.get('use_jerk', False)
        self.use_angles = config.get('use_angles', False)
        self.use_relative_angles = config.get('use_relative_angles', False)
        self.use_body_features = config.get('use_body_features', True)
        self.use_window_stats = config.get('use_window_stats', False)
        self.window_sizes = config.get('window_sizes', [5, 15, 30])

    def forward(self, x):
        features = []
        
        if self.use_distances:
            features.append(self.compute_distances(x))
            
        if self.use_velocity or self.use_acceleration or self.use_jerk:
            v, a, j = self.compute_derivatives(x)
            if self.use_velocity:
                features.append(v.reshape(v.shape[0], v.shape[1], -1))
            if self.use_acceleration:
                features.append(a.reshape(a.shape[0], a.shape[1], -1))
            if self.use_jerk:
                features.append(j.reshape(j.shape[0], j.shape[1], -1))
                
        if self.use_angles:
            features.append(self.compute_angles(x))
            
        if self.use_relative_angles:
            features.append(self.compute_relative_angles(x))

        if self.use_body_features:
            features.append(self.compute_body_features(x))
            
        return torch.cat(features, dim=-1)

    def compute_distances(self, keypoints):
        if keypoints.dim() == 5:
            B, T, M, K, C = keypoints.shape
            flat_kps = keypoints.view(B, T, M*K, C)
            diff = flat_kps.unsqueeze(3) - flat_kps.unsqueeze(2)
            dist_matrix = torch.norm(diff, dim=-1)
            num_points = dist_matrix.shape[2]
            triu_indices = torch.triu_indices(num_points, num_points, offset=1).to(keypoints.device)
            distances = dist_matrix[:, :, triu_indices[0], triu_indices[1]]
        else:
            if keypoints.dim() == 4:
                T, M, K, C = keypoints.shape
                flat_kps = keypoints.view(T, M*K, C)
            else:
                flat_kps = keypoints
            diff = flat_kps.unsqueeze(2) - flat_kps.unsqueeze(1)
            dist_matrix = torch.norm(diff, dim=-1)
            num_points = dist_matrix.shape[1]
            triu_indices = torch.triu_indices(num_points, num_points, offset=1).to(keypoints.device)
            distances = dist_matrix[:, triu_indices[0], triu_indices[1]]
        
        return distances

    def compute_derivatives(self, keypoints):
        velocity = torch.zeros_like(keypoints)
        velocity[..., 1:, :, :, :] = keypoints[..., 1:, :, :, :] - keypoints[..., :-1, :, :, :]
        
        acceleration = torch.zeros_like(velocity)
        acceleration[..., 1:, :, :, :] = velocity[..., 1:, :, :, :] - velocity[..., :-1, :, :, :]
        
        jerk = torch.zeros_like(acceleration)
        jerk[..., 1:, :, :, :] = acceleration[..., 1:, :, :, :] - acceleration[..., :-1, :, :, :]
        
        return velocity, acceleration, jerk

    def compute_angles(self, keypoints):
        nose = keypoints[..., 0, :]
        tail = keypoints[..., 6, :]
            
        vec = nose - tail
        angles = torch.atan2(vec[..., 1], vec[..., 0])
        
        return angles.unsqueeze(-1)

    def compute_body_features(self, keypoints):
        nose = keypoints[..., 0, :]
        neck = keypoints[..., 3, :]
        tail = keypoints[..., 6, :]
        
        body_length = torch.norm(nose - tail, dim=-1, keepdim=True)
        
        body_length_change = torch.zeros_like(body_length)
        body_length_change[..., 1:, :] = body_length[..., 1:, :] - body_length[..., :-1, :]
        
        v1 = nose - neck
        v2 = tail - neck
        
        v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-6)
        v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-6)
        
        cos_angle = (v1_norm * v2_norm).sum(dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
        curvature = torch.acos(cos_angle)
        
        B, T, M = keypoints.shape[:3]
        res = torch.cat([body_length, body_length_change, curvature], dim=-1)
        return res.view(B, T, -1)

    def compute_relative_angles(self, keypoints):
        nose = keypoints[..., 0, :]
        tail = keypoints[..., 3, :]
        orient_vec = nose - tail 
        orient_angle = torch.atan2(orient_vec[..., 1], orient_vec[..., 0]) 
        
        centroids = torch.mean(keypoints, dim=-2) 
        
        if keypoints.dim() == 5:
            B, T, M, K, C = keypoints.shape
            batch_dims = (B, T)
        else:
            T, M, K, C = keypoints.shape
            batch_dims = (T,)
            
        rel_angles_list = []
        
        for i in range(M):
            for j in range(M):
                if i == j: continue
                
                diff_angle = orient_angle[..., j] - orient_angle[..., i]
                diff_angle = torch.atan2(torch.sin(diff_angle), torch.cos(diff_angle))
                
                pos_vec = centroids[..., j, :] - centroids[..., i, :]
                pos_angle = torch.atan2(pos_vec[..., 1], pos_vec[..., 0])
                rel_pos_angle = pos_angle - orient_angle[..., i]
                rel_pos_angle = torch.atan2(torch.sin(rel_pos_angle), torch.cos(rel_pos_angle))
                
                rel_angles_list.append(diff_angle.unsqueeze(-1))
                rel_angles_list.append(rel_pos_angle.unsqueeze(-1))
                
        return torch.cat(rel_angles_list, dim=-1)

    def compute_window_stats(self, features):
        if features.dim() == 2:
            features = features.unsqueeze(0) 
            had_no_batch = True
        else:
            had_no_batch = False
            
        B, T, D = features.shape
        features_t = features.transpose(1, 2) 
        
        stats_list = []
        for w in self.window_sizes:
            padding = w // 2
            avg = torch.nn.functional.avg_pool1d(features_t, kernel_size=w, stride=1, padding=padding)
            max_val = torch.nn.functional.max_pool1d(features_t, kernel_size=w, stride=1, padding=padding)
            
            if avg.shape[2] > T:
                avg = avg[:, :, :T]
                max_val = max_val[:, :, :T]
            elif avg.shape[2] < T:
                avg = torch.nn.functional.pad(avg, (0, T - avg.shape[2]))
                max_val = torch.nn.functional.pad(max_val, (0, T - max_val.shape[2]))
                
            stats_list.append(avg.transpose(1, 2))
            stats_list.append(max_val.transpose(1, 2))
            
        out = torch.cat(stats_list, dim=-1)
        if had_no_batch:
            out = out.squeeze(0)
        return out

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
        if self.use_relative_angles:
            dim += num_mice * (num_mice - 1) * 2
            
        if self.use_body_features:
            dim += num_mice * 3
            
        if self.use_window_stats:
            base_dim = dim
            num_stats = 2 * len(self.window_sizes)
            dim += base_dim * num_stats
            
        return dim

    def __call__(self, keypoints):
        if keypoints.dim() == 5:
            B, T, M, K, C = keypoints.shape
            orig_shape_flat = (B, T, -1)
        else:
            T, M, K, C = keypoints.shape
            orig_shape_flat = (T, -1)
        
        if not self.config.get('enable_strong_features', True):
            return keypoints.view(*orig_shape_flat)
            
        feature_list = [keypoints.view(*orig_shape_flat)]
        
        if self.use_distances:
            feature_list.append(self.compute_distances(keypoints))
            
        velocity, acceleration, jerk = self.compute_derivatives(keypoints)
        
        if self.use_velocity:
            feature_list.append(velocity.view(*orig_shape_flat))
        if self.use_acceleration:
            feature_list.append(acceleration.view(*orig_shape_flat))
        if self.use_jerk:
            feature_list.append(jerk.view(*orig_shape_flat))
            
        if self.use_angles:
            feature_list.append(self.compute_angles(keypoints).view(*orig_shape_flat))
            
        if self.use_relative_angles:
            feature_list.append(self.compute_relative_angles(keypoints).view(*orig_shape_flat))

        if self.use_body_features:
            feature_list.append(self.compute_body_features(keypoints))
            
        combined_features = torch.cat(feature_list, dim=-1)
        
        if self.use_window_stats:
            window_stats = self.compute_window_stats(combined_features)
            combined_features = torch.cat([combined_features, window_stats], dim=-1)
            
        return combined_features
