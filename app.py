import gradio as gr
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/clean_hr_data.csv").reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
model = AutoModel.from_pretrained("intfloat/e5-small-v2")
model.eval()


def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state

    mask = inputs["attention_mask"].unsqueeze(-1)
    return ((tokens * mask).sum(1) / mask.sum(1)).numpy()


# Calculate candidate embeddings once
X_e5 = np.vstack([
    embed("passage: " + str(text))
    for text in df["clean_job_title"].fillna("")
])


def rank(query, starred):
    if not query.strip():
        return pd.DataFrame(columns=["job_title", "score"])

    scores = cosine_similarity(
        X_e5,
        embed("query: " + query.strip()),
    ).flatten()

    if starred:
        star_scores = cosine_similarity(
            X_e5,
            X_e5[starred],
        ).mean(axis=1)

        scores = 0.6 * scores + 0.4 * star_scores

    result = df[["job_title"]].copy()
    result["score"] = scores

    return result.sort_values("score", ascending=False).head(10)


def initial_search(query):
    top10 = rank(query, [])

    choices = [
        (f"⭐ {position + 1}: {row['job_title']}", int(index))
        for position, (index, row) in enumerate(top10.iterrows())
    ]

    return (
        top10.reset_index(drop=True),
        gr.update(choices=choices, value=[]),
        top10.reset_index(drop=True),
    )


with gr.Blocks() as demo:
    gr.Markdown("# HR Candidate Finder")

    query = gr.Textbox(label="Search candidates")
    button = gr.Button("Search")

    initial_results = gr.Dataframe(
        label="Initial top 10",
        interactive=False,
    )

    starred = gr.CheckboxGroup(
        choices=[],
        value=[],
        label="⭐ Star candidates from the initial top 10",
    )

    adjusted_results = gr.Dataframe(
        label="Adjusted top 10",
        interactive=False,
    )

    button.click(
        initial_search,
        inputs=query,
        outputs=[initial_results, starred, adjusted_results],
    )

    query.submit(
        initial_search,
        inputs=query,
        outputs=[initial_results, starred, adjusted_results],
    )

    starred.change(
        rank,
        inputs=[query, starred],
        outputs=adjusted_results,
    )


css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto;
}

h1 {
    color: #4338ca;
    text-align: center;
}

button {
    border-radius: 12px !important;
}

.block {
    border-radius: 12px !important;
}
"""





demo.launch(css=css)

