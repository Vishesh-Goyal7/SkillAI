import json
import torch
import torch.nn as nn
import joblib
import pandas as pd
from captum.attr import IntegratedGradients
from P1_predict_career import predict_top_careers
from P1_cluster_specialiser import ClusterSpecialiser  

with open("input.json") as f:
    user_data = json.load(f)

user_input = user_data["user_skills"]
user_df = pd.DataFrame([user_input])

predicted_jobs = predict_top_careers(user_df)
with open("results.json", "w") as f:
    json.dump({"recommended_jobs": predicted_jobs}, f, indent=2)

df = pd.read_csv("final_dataset_with_skill_clusters.csv")
job_to_cluster = df.drop_duplicates("label").set_index("label")["skill_cluster"].to_dict()

explanations = {}

for job in predicted_jobs:
    cluster_id = job_to_cluster.get(job)
    if cluster_id is None:
        continue

    model_path = f"cluster_specialisers/cluster_specialiser_{cluster_id}.pt"
    encoder_path = f"cluster_specialisers/job_label_encoder_{cluster_id}.pkl"

    job_encoder = joblib.load(encoder_path)
    output_dim = len(job_encoder.classes_)
    input_dim = user_df.shape[1]

    model = ClusterSpecialiser(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    try:
        target_idx = list(job_encoder.classes_).index(job)
    except ValueError:
        continue

    input_tensor = torch.tensor(user_df.values, dtype=torch.float32)
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_tensor, target=target_idx, return_convergence_delta=True)

    feature_scores = {
        feature: round(float(score), 4)
        for feature, score in zip(user_df.columns, attributions.detach().numpy().flatten())
        if float(score) != 0.0
    }

    sorted_scores = dict(sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    explanations[job] = sorted_scores

with open("explanations.json", "w") as f:
    json.dump(explanations, f, indent=2)

print("✅ Done! Saved recommended jobs → results.json and feature importances → explanations.json")