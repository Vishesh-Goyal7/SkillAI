import pandas as pd

df_profiles = pd.read_csv("career_profiles.csv")

df_clusters = pd.read_csv("clustered_job_titles.csv")

df_merged = df_profiles.merge(
    df_clusters,
    how="left",
    left_on="label",
    right_on="job_title"
)

df_merged.drop(columns=["job_title"], inplace=True)

df_merged.to_csv("final_dataset.csv", index=False)
print("✅ Cluster column successfully added and saved as 'final_dataset.csv'")