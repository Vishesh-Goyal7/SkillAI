import pandas as pd
import random
from collections import defaultdict

SKILLS_FILE = "../Dataset/Skills.txt"
OCCUPATION_FILE = "../Dataset/Occupation_Data.txt"
PROFILES_PER_ROLE = 50
MIN_SKILLS_REQUIRED = 3

skills_df = pd.read_csv(SKILLS_FILE, sep="\t", encoding="latin-1")
occupations_df = pd.read_csv(OCCUPATION_FILE, sep="\t", header=None, names=["O*NET-SOC Code", "Title", "Description"])

skills_df = skills_df[(skills_df["Scale ID"] == "IM") & (skills_df["Data Value"] >= 3.0)]
skills_df["O*NET-SOC Code"] = skills_df["O*NET-SOC Code"].astype(str).str.strip()
occupations_df["O*NET-SOC Code"] = occupations_df["O*NET-SOC Code"].astype(str).str.strip()
occupations_df["Title"] = occupations_df["Title"].astype(str).str.strip()

soc_to_title = dict(zip(occupations_df["O*NET-SOC Code"], occupations_df["Title"]))

def infer_education_level(job_title):
    title = job_title.lower()
    if any(k in title for k in ["scientist", "research", "physicist", "economist"]):
        return ["PhD", "Masters"]
    elif any(k in title for k in ["engineer", "developer", "analyst", "manager", "consultant", "technologist", "specialist", "architect"]):
        return ["Bachelors", "Masters"]
    elif any(k in title for k in ["assistant", "technician", "clerk", "drafter", "support"]):
        return ["Diploma", "Bachelors"]
    else:
        return ["Bachelors"]

occupation_skills = defaultdict(list)
for _, row in skills_df.iterrows():
    soc = row["O*NET-SOC Code"]
    skill = row["Element Name"]
    job_title = soc_to_title.get(soc, soc)  
    occupation_skills[job_title].append(skill)

all_skills = list(set(skills_df["Element Name"]))
profiles = []

for job_title, skills in occupation_skills.items():
    if len(skills) < MIN_SKILLS_REQUIRED:
        continue

    top_skills = list(set(skills))[:10]
    edu_pool = infer_education_level(job_title)

    for _ in range(PROFILES_PER_ROLE):
        selected_skills = set(random.sample(top_skills, k=min(5, len(top_skills))))
        noise_skills = set(random.sample(all_skills, k=3)) - selected_skills
        all_user_skills = list(selected_skills.union(noise_skills))

        record = {skill: 1 for skill in all_user_skills}

        edu = random.choice(edu_pool)
        record.update({f"edu_{level}": int(level == edu) for level in ["Diploma", "Bachelors", "Masters", "PhD"]})

        record["label"] = job_title
        profiles.append(record)

if profiles:
    df = pd.DataFrame(profiles).fillna(0)
    df.to_csv("../Phase1_Model_Creation/career_profiles.csv", index=False)
    print(f"✅ Total profiles: {len(df)}  |  Columns: {len(df.columns)}")
else:
    print("⚠️ No profiles generated. Try lowering MIN_SKILLS_REQUIRED or check data files.")