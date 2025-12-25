import pandas as pd
import os
import numpy as np

file_path = r"c:\Users\Windows11\Downloads\mice_social_action_new\data\train_tracking\AdaptableSnail\44566106.parquet"
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print("Dtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head())
    
    # Check if values are numeric
    try:
        vals = df.values
        print(f"\nValues shape: {vals.shape}")
        print(f"Values dtype: {vals.dtype}")
        np.isnan(vals)
        print("np.isnan check passed")
    except Exception as e:
        print(f"np.isnan check failed: {e}")
else:
    print("File not found")
