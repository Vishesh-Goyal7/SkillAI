import pandas as pd
import numpy as np
from P1_predict_career import predict_top_careers
from sklearn.utils import shuffle

DATA_PATH = "final_dataset_with_skill_clusters.csv"
N_SAMPLES = 100
TOP_K_JOBS = 10
TOP_K_CLUSTERS = 4

df = pd.read_csv(DATA_PATH)
df = shuffle(df, random_state=53).reset_index(drop=True)

correct_count = 0
samples_checked = 0

for i in range(N_SAMPLES):
    row = df.iloc[i]
    true_job = row["label"]
    input_features = row.drop(labels=["label", "skill_cluster"]).to_dict()

    try:
        predictions = predict_top_careers(input_features, top_k_clusters=TOP_K_CLUSTERS, top_k_jobs=TOP_K_JOBS)
        samples_checked += 1
        if true_job in predictions:
            correct_count += 1
        print(f"✔️ Sample {i+1}: Actual = {true_job} | Predicted = {predictions} | {'✔️' if true_job in predictions else '❌'}")
    except Exception as e:
        print(f"⚠️ Error on sample {i+1}: {e}")
        continue

accuracy = correct_count / samples_checked if samples_checked > 0 else 0
print(f"\n🎯 Top-{TOP_K_JOBS * TOP_K_CLUSTERS} Job Accuracy: {accuracy * 100:.2f}%  ({correct_count}/{samples_checked})")
