import pandas as pd
import random
import json

file_path = 'final_dataset_with_skill_clusters.csv' 

drop_cols = ['skill_cluster', 'label']  

df = pd.read_csv(file_path)

random_row = df.sample(n=12).reset_index(drop=True)

dropped_values = random_row[drop_cols].iloc[0].to_dict()

row_dropped = random_row.drop(columns=drop_cols)

json_data = {"user_skills":{col: int(val) for col, val in row_dropped.iloc[0].items()}}

with open('input.json', 'w') as f:
    json.dump(json_data, f, indent=4)

print("Dropped cluster values:")
for cluster, value in dropped_values.items():
    print(f"{cluster}: {value}")