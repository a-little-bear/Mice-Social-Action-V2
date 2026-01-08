import pandas as pd
import numpy as np
import glob
import os

def analyze_density():
    # Path to annotations
    base_path = r"d:\Projects\mice_social_action_new\data\train_annotation"
    
    # Get all parquet files
    files = glob.glob(os.path.join(base_path, "*", "*.parquet"))
    
    print(f"Found {len(files)} annotation files.")
    
    total_frames = 0
    total_positives = 0
    
    # Sample a subset to save time if there are too many
    # But for accurate stats, we should try to read many. Let's read first 20 files.
    sample_files = files[:20] 
    
    print(f"Analyzing {len(sample_files)} files...")
    
    all_positive_counts = []
    
    for f in sample_files:
        try:
            df = pd.read_parquet(f)
            # Assuming columns are the classes
            # Values are 0 or 1
            # We treat any class=1 as a positive event for that frame? 
            # Or is this per-class?
            # pos_weight is usually applied per-class in multi-label BCE. Is pos_weight scalar or vector?
            # The config used scalar 12.0 for all classes.
            
            # Let's count total 1s vs total 0s across all classes
            # (or average positivity rate)
            
            vals = df.values
            frames, classes = vals.shape
            
            pos_count = np.sum(vals)
            total_elements = frames * classes
            
            total_frames += total_elements # Counting element-wise
            total_positives += pos_count
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if total_frames == 0:
        print("No data read.")
        return

    pos_rate = total_positives / total_frames
    neg_rate = 1 - pos_rate
    
    print(f"Total entries (frames * classes): {total_frames}")
    print(f"Total positives: {total_positives}")
    print(f"Positive Rate: {pos_rate:.6f}")
    
    if pos_rate > 0:
        suggested_weight = neg_rate / pos_rate
        print(f"Suggested pos_weight (neg/pos): {suggested_weight:.2f}")
    else:
        print("Pos_rate is 0!")

if __name__ == "__main__":
    analyze_density()
