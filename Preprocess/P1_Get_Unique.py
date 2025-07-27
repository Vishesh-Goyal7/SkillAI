import pandas as pd

df = pd.read_csv("../backend/final_dataset_with_skill_clusters.csv")
df = df["label"]

jobs = []

for i in df:
    if i in jobs:
        continue
    else : 
        jobs.append(i)

df = pd.DataFrame(jobs)
df.to_csv("../Dataset/unique_job_list.csv", index=False)