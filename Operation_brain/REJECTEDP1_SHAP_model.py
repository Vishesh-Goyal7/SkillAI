import pandas as pd
# import numpy as np
# import shap
# import joblib
# from xgboost import XGBClassifier

# model = joblib.load("cluster_model_xgb.pkl")
# df = pd.read_csv("final_dataset.csv")

# X = df.drop(columns=["label", "cluster"])

# explainer = shap.Explainer(model)
# shap_values = explainer(X)
# shap_array = shap_values.values 
# mean_abs_shap = np.abs(shap_array).mean(axis=(0, 2))

# shap_df = pd.DataFrame({
#     "feature": X.columns,
#     "mean_abs_shap": mean_abs_shap
# }).sort_values(by="mean_abs_shap", ascending=False)

# shap_df_top = shap_df[shap_df["mean_abs_shap"] >= 0.02]
# print("🔥 Keeping", len(shap_df_top), "features with SHAP ≥ 0.02")

# shap_df_top.to_csv("shap_feature_importance.csv", index=False)
# print("📊 Top Features:\n")
# print(shap_df)



df = pd.read_csv("final_dataset.csv")

shap_df = pd.read_csv("shap_feature_importance.csv")

top_features = shap_df[shap_df["mean_abs_shap"] >= 0.02]["feature"].tolist()

columns_to_keep = top_features + ["label", "cluster"]

df_filtered = df[columns_to_keep]

df_filtered.to_csv("final_dataset.csv", index=False)
print(f"✅ Filtered dataset saved with {df_filtered.shape[1]} columns and {df_filtered.shape[0]} rows.")