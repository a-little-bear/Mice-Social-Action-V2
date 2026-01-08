import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
import ast
from concurrent.futures import ThreadPoolExecutor
from .transforms import CoordinateTransform, Augmentation, FPSCorrection, BodyPartMapping
from .features import FeatureGenerator
from .sampling import ActionRichSampler

class MABeDataset(Dataset):
    _video_cache = {}
    _label_full_cache = {} # New: Compact global cache for full-video labels

    def __init__(self, data_path, config, mode='train'):
        self.data_path = data_path
        self.config = config
        self.mode = mode
        self.window_size = config['data'].get('window_size', 512)
        self.stride = self.window_size
        
        self.fps_correction = FPSCorrection(
            target_fps=config['data']['preprocessing']['target_fps'],
            target_length=self.window_size
        )
        self.coord_transform = CoordinateTransform(
            view=config['data']['preprocessing']['view']
        )
        self.augmentor = Augmentation(config['data']['augmentation']) if mode == 'train' else None
        self.feature_generator = FeatureGenerator(config['data']['features'])
        self.body_part_mapping = BodyPartMapping(enabled=config['data']['preprocessing'].get('unify_body_parts', False))
        
        self.data = []
        self.labels = []
        self.class_weights = None
        self.cache_size = config['data'].get('cache_size', 128)
        self.preload = config['data'].get('preload', False)
        
        csv_name = 'train.csv' if mode in ['train', 'val'] else 'test.csv'
        csv_path = os.path.join(data_path, csv_name)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            self.classes = []
            self.class_to_idx = {}
            self.num_classes = 0
        else:
            df = pd.read_csv(csv_path)
            
            if 'behaviors_labeled' in df.columns:
                behaviors_str = df.iloc[0]['behaviors_labeled']
                try:
                    self.classes = json.loads(behaviors_str)
                except:
                    try:
                        self.classes = ast.literal_eval(behaviors_str)
                    except:
                        self.classes = []
                
                self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
                self.num_classes = len(self.classes)
            else:
                self.classes = []
                self.class_to_idx = {}
                self.num_classes = 0
            
            use_sampler = mode == 'train' and config['data']['sampling']['strategy'] == 'action_rich'
            
            for idx, row in df.iterrows():
                lab_id = row['lab_id']
                video_id = str(row['video_id'])
                
                tracking_dir = 'train_tracking' if mode in ['train', 'val'] else 'test_tracking'
                annotation_dir = 'train_annotation' if mode in ['train', 'val'] else 'test_annotation'
                
                tracking_file = os.path.join(data_path, tracking_dir, lab_id, f'{video_id}.parquet')
                annotation_file = os.path.join(data_path, annotation_dir, lab_id, f'{video_id}.parquet')
                
                if not os.path.exists(tracking_file):
                    continue
                
                anno_df = pd.DataFrame()
                if use_sampler and os.path.exists(annotation_file):
                    try:
                        anno_df = pd.read_parquet(annotation_file)
                    except:
                        pass
                
                if 'video_duration_sec' in row and 'frames_per_second' in row:
                    T_est = int(row['video_duration_sec'] * row['frames_per_second'])
                    fps = row['frames_per_second']
                else:
                    T_est = 18000
                    fps = 30.0
                
                target_fps = config['data']['preprocessing']['target_fps']
                fps_ratio = fps / target_fps
                adj_window = int(self.window_size * fps_ratio)
                adj_stride = int(self.stride * fps_ratio)
                
                adj_window = max(1, adj_window)
                adj_stride = max(1, adj_stride)

                num_windows = (T_est + adj_stride - 1) // adj_stride
                
                for w in range(num_windows):
                    start = w * adj_stride
                    end = start + adj_window
                    
                    self.data.append({
                        'tracking_path': tracking_file,
                        'annotation_path': annotation_file,
                        'lab_id': lab_id,
                        'video_id': video_id,
                        'subject_id': 0,
                        'start': start,
                        'end': end,
                        'fps': fps
                    })
                    
                    if use_sampler:
                        has_action = False
                        if not anno_df.empty:
                            overlaps = anno_df[
                                (anno_df['start_frame'] < end) & 
                                (anno_df['stop_frame'] > start)
                            ]
                            has_action = not overlaps.empty
                        self.labels.append(has_action)

                if use_sampler and not anno_df.empty:
                    for _, row in anno_df.iterrows():
                        event_start = row['start_frame']
                        event_stop = row['stop_frame']
                        event_center = (event_start + event_stop) // 2
                        
                        start = max(0, event_center - adj_window // 2)
                        end = start + adj_window
                        
                        if start >= T_est:
                            continue
                            
                        self.data.append({
                            'tracking_path': tracking_file,
                            'annotation_path': annotation_file,
                            'lab_id': lab_id,
                            'video_id': video_id,
                            'subject_id': 0,
                            'start': start,
                            'end': end,
                            'fps': fps
                        })
                        self.labels.append(True) 

        if self.preload:
            if len(MABeDataset._video_cache) > 0:
                print(f"Data already preloaded in memory. Skipping redundant preload for mode: {mode}")
            else:
                print(f"Starting parallel preload of all tracking data & labels into RAM (Mode: {mode})...")
                
                # Get unique video/label pairs to load
                unique_keys = set()
                preload_args = []
                for d in self.data:
                    key = (d['tracking_path'], d['lab_id'], d['video_id'], d['annotation_path'])
                    if key not in unique_keys:
                        unique_keys.add(key)
                        preload_args.append(key)
                
                if not preload_args:
                    print(f"No data found to preload for mode: {mode}")
                else:
                    from tqdm import tqdm
                    max_workers = min(len(preload_args), 16)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        def _preload_task(x):
                            # x = (tracking_path, lab_id, video_id, annotation_path)
                            kps = self._load_video(x[0], x[1], x[2])
                            if kps is not None and x[3] and os.path.exists(x[3]):
                                self._load_full_labels(x[3], x[2], kps.shape[0])
                            return True
                        
                        list(tqdm(executor.map(_preload_task, preload_args), 
                                  total=len(preload_args), desc="Preloading Data"))
                    
                    print(f"Preloaded {len(MABeDataset._video_cache)} videos and {len(MABeDataset._label_full_cache)} label tensors.")
                    
                    # New: Calculate global class frequencies for 'new_focal' loss
                    print("Calculating global class frequencies for optimized loss...")
                    all_counts = torch.zeros(self.num_classes, dtype=torch.float64)
                    for lbl in MABeDataset._label_full_cache.values():
                        all_counts += lbl.sum(dim=0).to(torch.float64)
                    
                    self.class_counts = all_counts
                    total_samples = all_counts.sum()
                    if total_samples > 0:
                        # Use a more stable weighting (log-inverse frequency or effective number of samples)
                        # We'll use smooth inverse frequency: W = log(1 + total / (count + 1))
                        # This avoids extreme weights while still emphasizing rare classes.
                        weights = torch.log1p(total_samples / (all_counts + 1.0))
                        
                        # Normalize so mean weight is 1.0
                        self.class_weights = (weights / weights.mean()).to(torch.float32)
                        print(f"Computed weights for {self.num_classes} classes. Max weight: {self.class_weights.max().item():.2f}")
                    else:
                        self.class_weights = torch.ones(self.num_classes, dtype=torch.float32)
                    
                    import gc
                    gc.collect()

        if mode == 'train' and config['data']['sampling']['strategy'] == 'action_rich':
            self.sampler = ActionRichSampler(self.labels, window_size=512, bias_factor=config['data']['sampling']['bias_factor'])
        else:
            self.sampler = None

    def __len__(self):
        return len(self.data)

    def _load_full_labels(self, annotation_path, video_id, total_frames):
        """Loads all annotations for a video and stores them in a compact uint8 tensor."""
        if annotation_path in MABeDataset._label_full_cache:
            return MABeDataset._label_full_cache[annotation_path]
            
        labels = torch.zeros((total_frames, self.num_classes), dtype=torch.uint8)
        if not os.path.exists(annotation_path):
            return labels
            
        try:
            df = pd.read_parquet(annotation_path)
            for _, row in df.iterrows():
                try:
                    agent_id = int(row['agent_id'])
                    target_id = int(row['target_id'])
                    action = row['action']
                    
                    subject = f"mouse{agent_id + 1}"
                    obj = "self" if agent_id == target_id else f"mouse{target_id + 1}"
                    composite_action = f"{subject},{obj},{action}"
                except (ValueError, KeyError):
                    composite_action = row['action'] if 'action' in row else None

                target_class = None
                if composite_action in self.class_to_idx:
                    target_class = composite_action
                elif 'action' in row and row['action'] in self.class_to_idx:
                    target_class = row['action']
                
                if target_class:
                    idx = self.class_to_idx[target_class]
                    s = max(0, int(row['start_frame']))
                    e = min(total_frames, int(row['stop_frame']))
                    if s < e:
                        labels[s:e, idx] = 1
            
            # Use compact representation
            MABeDataset._label_full_cache[annotation_path] = labels
            return labels
        except Exception as e:
            print(f"Error processing labels for {annotation_path}: {e}")
            return labels

    def _create_label_tensor(self, annotation_path, start_frame, end_frame, num_frames):
        """Now uses the pre-loaded full label tensor and slices it."""
        if annotation_path not in MABeDataset._label_full_cache:
            # Fallback for dynamic loading if not preloaded
            # We need to know the total frames; if not preloaded, we estimate or use end_frame
            self._load_full_labels(annotation_path, None, end_frame)
            
        full_labels = MABeDataset._label_full_cache.get(annotation_path)
        if full_labels is None:
            return torch.zeros((num_frames, self.num_classes), dtype=torch.float32)

        # Slice the relevant part
        # Note: If frames are corrected/resampled, we need to handle that.
        # But for 'target_fps', this slice logic should match the keypoints slicing.
        original_len = end_frame - start_frame
        
        # Initial slice (raw frames)
        s_idx = max(0, start_frame)
        e_idx = min(full_labels.shape[0], end_frame)
        
        label_slice = full_labels[s_idx:e_idx]
        
        # If we need padding
        if label_slice.shape[0] < original_len:
            pad_len = original_len - label_slice.shape[0]
            padding = torch.zeros((pad_len, self.num_classes), dtype=torch.uint8)
            label_slice = torch.cat([label_slice, padding], dim=0)
            
        # Resample to match num_frames (target_fps)
        if label_slice.shape[0] != num_frames:
            # Simple nearest neighbor resampling for labels to preserve 0/1
            indices = torch.linspace(0, label_slice.shape[0]-1, num_frames).long()
            label_slice = label_slice[indices]
            
        return label_slice.float()

    def _load_video(self, tracking_path, lab_id, video_id=None):
        cache_key = (tracking_path, lab_id)
        if cache_key in MABeDataset._video_cache:
            return MABeDataset._video_cache[cache_key]
            
        try:
            df = pd.read_parquet(tracking_path)
            
            expected_parts = self.body_part_mapping.lab_configs.get(lab_id, [])
            if not expected_parts:
                expected_parts = sorted(df['bodypart'].unique())
            
            mice = df['mouse_id'].unique()
            mice.sort()
            
            target_M = 2
            if len(mice) > target_M:
                mice = mice[:target_M]
            
            max_frame = df['video_frame'].max()
            T = max_frame + 1
            M = target_M
            P = len(expected_parts)
            
            keypoints = np.full((T, M, P, 2), np.nan, dtype=np.float32)
            
            swapped_videos = ['1212811043', '1260392287', '1351098077'] 
            if lab_id == 'AdaptableSnail' and video_id in swapped_videos and len(mice) >= 2:
                mouse_map = {mice[0]: 1, mice[1]: 0}
                for i in range(2, len(mice)):
                    mouse_map[mice[i]] = i
            else:
                mouse_map = {m: i for i, m in enumerate(mice)}
            
            part_map = {p: i for i, p in enumerate(expected_parts)}
            
            df = df[df['mouse_id'].isin(mice)]
            df = df[df['bodypart'].isin(expected_parts)]
            
            f_idx = df['video_frame'].values
            m_idx = df['mouse_id'].map(mouse_map).values
            p_idx = df['bodypart'].map(part_map).values
            
            keypoints[f_idx, m_idx, p_idx, 0] = df['x'].values
            keypoints[f_idx, m_idx, p_idx, 1] = df['y'].values
            
            # Memory Cleanup: Free the large dataframe as soon as data is copied
            del df
            import gc
            gc.collect()
            
            if self.config['data']['preprocessing'].get('interpolate_nans', True):
                mask = np.isnan(keypoints)
                if mask.any():
                    T_dim, M_dim, P_dim, C_dim = keypoints.shape
                    flat = keypoints.reshape(T_dim, -1)
                    df_interp = pd.DataFrame(flat)
                    df_interp = df_interp.interpolate(method='linear', limit_direction='both')
                    keypoints = df_interp.values.reshape(keypoints.shape)
                    keypoints = np.nan_to_num(keypoints)

            if not self.preload and len(MABeDataset._video_cache) >= self.cache_size:
                it = iter(MABeDataset._video_cache)
                try:
                    MABeDataset._video_cache.pop(next(it))
                except (StopIteration, KeyError):
                    pass
            
            keypoints = torch.from_numpy(keypoints).to(torch.bfloat16)
            MABeDataset._video_cache[cache_key] = keypoints
            
            return keypoints
        except Exception as e:
            print(f"Error loading {tracking_path}: {e}")
            return None

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        tracking_path = sample_info['tracking_path']
        lab_id = sample_info['lab_id']
        video_id = sample_info['video_id']
        subject_id = sample_info['subject_id']
        start = sample_info['start']
        end = sample_info['end']
        fps = sample_info.get('fps', 30.0)
        
        keypoints_full = self._load_video(tracking_path, lab_id, video_id)
        
        if keypoints_full is None:
             dummy_kps = torch.zeros(self.window_size, 2, 7, 2)
             if self.mode in ['train', 'val']:
                 dummy_labels = torch.zeros(self.window_size, self.num_classes)
                 return dummy_kps, dummy_labels, lab_id, subject_id, video_id
             return dummy_kps, lab_id, subject_id, video_id
             
        T_full = keypoints_full.shape[0]
        
        if start >= T_full:
            keypoints = np.zeros((self.window_size, keypoints_full.shape[1], keypoints_full.shape[2], 2), dtype=np.float32)
        else:
            actual_end = min(end, T_full)
            
            if isinstance(keypoints_full, torch.Tensor):
                keypoints_slice = keypoints_full[start:actual_end].to(torch.float32).numpy()
            else:
                keypoints_slice = keypoints_full[start:actual_end]
            
            if actual_end < end:
                pad_len = end - actual_end
                padding = np.zeros((pad_len, keypoints_full.shape[1], keypoints_full.shape[2], 2), dtype=np.float32)
                keypoints = np.concatenate([keypoints_slice, padding], axis=0)
            else:
                keypoints = keypoints_slice

        if self.config['data']['preprocessing']['fix_fps']:
            keypoints = self.fps_correction(keypoints, lab_id, current_fps=fps)

        keypoints = self.body_part_mapping(keypoints, lab_id)

        keypoints = self.coord_transform(keypoints)
        
        if self.augmentor:
            keypoints = self.augmentor(keypoints)
            
        keypoints_tensor = torch.FloatTensor(keypoints)
        
        if self.mode in ['train', 'val']:
            label = self._create_label_tensor(
                sample_info['annotation_path'], 
                start, 
                end, 
                keypoints_tensor.shape[0]
            )
            
            return keypoints_tensor, label, lab_id, subject_id, video_id
            
        return keypoints_tensor, lab_id, subject_id, video_id
