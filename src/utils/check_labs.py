import pandas as pd
import os
from src.data.transforms import BodyPartMapping

csv_path = r"c:\Users\Windows11\Downloads\mice_social_action_new\data\train.csv"
df = pd.read_csv(csv_path)
unique_labs = df['lab_id'].unique()

mapper = BodyPartMapping(enabled=True)
known_labs = mapper.lab_configs.keys()

print("Unique labs in CSV:", unique_labs)
print("\nMissing labs in Mapper:")
for lab in unique_labs:
    if lab not in known_labs:
        print(lab)
