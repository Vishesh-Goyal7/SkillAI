import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("combined_career_profiles.csv")

X = df.drop(columns=["label"])
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, "career_model_xgb.pkl")
print("💾 Model and label encoder saved.")
