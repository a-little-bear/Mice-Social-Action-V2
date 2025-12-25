import pandas as pd
import os

file_path = r"c:\Users\Windows11\Downloads\mice_social_action_new\data\train_annotation\AdaptableSnail\44566106.parquet"
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print(df.head())
else:
    print("File not found")
