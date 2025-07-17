import pandas as pd

df = pd.read_csv("../Phase1_Model_Creation/career_profiles.csv")
df = df["label"]

jobs = []

for i in df:
    if i in jobs:
        continue
    else : 
        jobs.append(i)

df = pd.DataFrame(jobs)
df.to_csv("../Dataset/unique_job_list.csv", index=False)