import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

INPUT_CSV = "career_profiles.csv"
OUTPUT_CSV = "final_dataset_with_skill_clusters.csv"
N_CLUSTERS = 7  

df = pd.read_csv(INPUT_CSV)

feature_cols = [col for col in df.columns if col.startswith("edu_") or col in ["label", "cluster"]]
skill_cols = [col for col in df.columns if col not in feature_cols]

job_skill_matrix = df.groupby("label")[skill_cols].mean()

X = job_skill_matrix.values

X_norm = X / (X**2).sum(axis=1, keepdims=True)**0.5

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
job_clusters = kmeans.fit_predict(X_norm)

job_skill_matrix["skill_cluster"] = job_clusters
job_skill_matrix.reset_index(inplace=True)

df = df.merge(job_skill_matrix[["label", "skill_cluster"]], on="label", how="left")
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Skill-based job clusters created and saved to {OUTPUT_CSV}")
print(f"🧠 Jobs are now clustered by skills into {N_CLUSTERS} groups.")
