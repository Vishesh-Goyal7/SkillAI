import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

xgb_model = joblib.load("cluster_model_xgb.pkl")
cluster_encoder = joblib.load("cluster_label_encoder.pkl")

class ClusterSpecialiser(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def predict_top_careers(user_input: pd.DataFrame, top_k_clusters=5, top_k_jobs=2):
    if isinstance(user_input, dict):
        user_input = pd.DataFrame([user_input])
    
    cluster_probs = xgb_model.predict_proba(user_input.values)
    top_cluster_indices = np.argsort(cluster_probs[0])[::-1][:top_k_clusters]
    top_clusters = cluster_encoder.inverse_transform(top_cluster_indices)
    
    final_recommendations = []

    for cluster_id in top_clusters:
        cluster_id_str = str(cluster_id)

        job_encoder_path = f"cluster_specialisers/job_label_encoder_{cluster_id_str}.pkl"
        dnn_path = f"cluster_specialisers/cluster_specialiser_{cluster_id_str}.pt"

        try:
            job_encoder = joblib.load(job_encoder_path)
        except FileNotFoundError:
            print(f"⚠️ No encoder found for cluster {cluster_id}")
            continue

        input_dim = user_input.shape[1]
        output_dim = len(job_encoder.classes_)

        dnn = ClusterSpecialiser(input_dim, output_dim)
        dnn.load_state_dict(torch.load(dnn_path))
        dnn.eval()

        with torch.no_grad():
            input_tensor = torch.tensor(user_input.values, dtype=torch.float32)
            job_logits = dnn(input_tensor).numpy().flatten()
            top_job_indices = np.argsort(job_logits)[::-1][:top_k_jobs]
            top_jobs = job_encoder.inverse_transform(top_job_indices)

            final_recommendations.extend(list(top_jobs))

    return final_recommendations