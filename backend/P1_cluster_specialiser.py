import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DATA_PATH = "final_dataset_with_skill_clusters.csv"
SAVE_DIR = "cluster_specialisers/"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

clusters = df["skill_cluster"].unique()

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

def main():
    for cluster_id in clusters:
        print(f"\n🔧 Training model for Cluster {cluster_id}...")

        df_c = df[df["skill_cluster"] == cluster_id]
        X = df_c.drop(columns=["label", "skill_cluster"])
        y = df_c["label"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

        model = ClusterSpecialiser(X_train.shape[1], len(le.classes_)).to("cpu")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(30):
            model.train()
            for xb, yb in loader:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"{SAVE_DIR}cluster_specialiser_{cluster_id}.pt")
        joblib.dump(le, f"{SAVE_DIR}job_label_encoder_{cluster_id}.pkl")
        print(f"✅ Model & encoder saved for Cluster {cluster_id}")

if __name__ == "__main__":
    main()