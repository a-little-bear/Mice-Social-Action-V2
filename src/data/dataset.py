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
    _anno_cache = {}

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
                print(f"Starting parallel preload of all tracking data into RAM (Mode: {mode})...")
                unique_videos = {}
                for d in self.data:
                    key = (d['tracking_path'], d['lab_id'], d['video_id'])
                    unique_videos[key] = True
                
                from tqdm import tqdm
                video_list = list(unique_videos.keys())
                
                if not video_list:
                    print(f"No videos found to preload for mode: {mode}")
                else:
                    max_workers = min(len(video_list), 16)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        def _preload_task(x):
                            self._load_video(*x)
                            anno_path = os.path.join(self.data_path, 'train_annotation', x[1], f'{x[2]}.parquet')
                            if os.path.exists(anno_path):
                                try:
                                    MABeDataset._anno_cache[anno_path] = pd.read_parquet(anno_path)
                                except:
                                    pass
                        
                        list(tqdm(executor.map(_preload_task, video_list), 
                                  total=len(video_list), desc="Preloading Videos & Annotations"))
                    
                    print(f"Preloaded {len(MABeDataset._video_cache)} videos and {len(MABeDataset._anno_cache)} annotations into RAM.")

        if mode == 'train' and config['data']['sampling']['strategy'] == 'action_rich':
            self.sampler = ActionRichSampler(self.labels, window_size=512, bias_factor=config['data']['sampling']['bias_factor'])
        else:
            self.sampler = None

    def __len__(self):
        return len(self.data)

    def _create_label_tensor(self, annotation_path, start_frame, end_frame, num_frames):
        labels = torch.zeros((num_frames, self.num_classes), dtype=torch.float32)
        if not os.path.exists(annotation_path):
            return labels
            
        original_len = end_frame - start_frame
        scale = num_frames / original_len if original_len > 0 else 1.0
        
        if annotation_path in MABeDataset._anno_cache:
            df = MABeDataset._anno_cache[annotation_path]
        else:
            try:
                df = pd.read_parquet(annotation_path)
                if not self.preload and len(MABeDataset._anno_cache) >= self.cache_size:
                    it = iter(MABeDataset._anno_cache)
                    try:
                        MABeDataset._anno_cache.pop(next(it))
                    except (StopIteration, KeyError):
                        pass
                MABeDataset._anno_cache[annotation_path] = df
            except:
                return labels
            
        df_slice = df[
            (df['start_frame'] < end_frame) & 
            (df['stop_frame'] > start_frame)
        ]
        
        for _, row in df_slice.iterrows():
            try:
                agent_id = int(row['agent_id'])
                target_id = int(row['target_id'])
                action = row['action']
                
                subject = f"mouse{agent_id + 1}"
                if agent_id == target_id:
                    obj = "self"
                else:
                    obj = f"mouse{target_id + 1}"
                    
                composite_action = f"{subject},{obj},{action}"
            except (ValueError, KeyError):
                composite_action = row['action'] if 'action' in row else None

            s = row['start_frame']
            e = row['stop_frame']
            
            target_class = None
            if composite_action in self.class_to_idx:
                target_class = composite_action
            elif 'action' in row and row['action'] in self.class_to_idx:
                target_class = row['action']
            
            if target_class:
                idx = self.class_to_idx[target_class]
                s_w = max(0, int((s - start_frame) * scale))
                e_w = min(num_frames, int((e - start_frame) * scale))
                
                if s_w < e_w:
                    labels[s_w:e_w, idx] = 1.0
                             
        return labels

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
