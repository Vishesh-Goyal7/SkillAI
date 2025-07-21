import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("../Dataset/unique_job_list.csv")
df.columns = ['job_title']
job_titles = df['job_title'].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(job_titles, show_progress_bar=True)

num_clusters = 7
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

df.to_csv("clustered_job_titles.csv", index=False)
