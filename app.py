import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gradio as gr
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

data_path = "data/clean_hr_data.csv"
df = pd.read_csv(data_path)

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
e5 = AutoModel.from_pretrained("intfloat/e5-small-v2")
e5.eval()


def get_title_vector_e5(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = e5(**inputs)

    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()

    vector = (
        (token_embeddings * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )

    return vector.squeeze(0).numpy()


# Calculate candidate embeddings once when the app starts
print("Preparing candidate embeddings...")

X_e5 = np.vstack([
    get_title_vector_e5("passage: " + str(text))
    for text in df["clean_job_title"].fillna("")
])

print("Candidates ready!")


def search(query):
    if not query.strip():
        return pd.DataFrame(columns=["job_title", "similarity"])

    # Embed whatever the user enters
    query_vector = get_title_vector_e5("query: " + query.strip())

    scores = cosine_similarity(
        query_vector.reshape(1, -1),
        X_e5,
    ).flatten()

    ranked = df[["job_title"]].copy()
    ranked["similarity"] = scores

    return (
        ranked.sort_values("similarity", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(
        label="Search candidates",
        placeholder="aspiring human resources",
    ),
    outputs=gr.Dataframe(label="Top 10 candidates"),
    title="HR Candidate Ranking",
)

demo.launch(share=True)