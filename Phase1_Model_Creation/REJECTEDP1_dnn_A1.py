import pandas as pd
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "final_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label", "cluster"])
y = df["cluster"].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

joblib.dump(le, "cluster_label_encoder_dnn.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

class ClusterDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = ClusterDNN(X_train.shape[1], num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test.to(DEVICE)).argmax(dim=1).cpu().numpy()
    print("\n✅ Accuracy:", accuracy_score(y_test, preds))
    print("\n📊 Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

torch.save(model.state_dict(), "cluster_dnn_model.pt")
print("💾 DNN model saved as 'cluster_dnn_model.pt'")