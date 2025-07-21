import pandas as pd
import random
import json

file_path = 'final_dataset_with_skill_clusters.csv'  # Change as needed

# Columns to drop
drop_cols = ['skill_cluster', 'label']  # Replace with your actual column names

# Load dataset
df = pd.read_csv(file_path)

# Randomly select one row
random_row = df.sample(n=12).reset_index(drop=True)

# Log the values of the two dropped columns
dropped_values = random_row[drop_cols].iloc[0].to_dict()

# Drop specified columns
row_dropped = random_row.drop(columns=drop_cols)

# Create JSON object from columns where value == 1
json_data = {col: int(val) for col, val in row_dropped.iloc[0].items()}

# Save to input.json
with open('input.json', 'w') as f:
    json.dump(json_data, f, indent=4)

# Print dropped cluster values
print("Dropped cluster values:")
for cluster, value in dropped_values.items():
    print(f"{cluster}: {value}")