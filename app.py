import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

data_path = "data/clean_hr_data.csv"
df = pd.read_csv(data_path)

alpha = 0.6

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


print("Preparing candidate embeddings...")

X_e5 = np.vstack([
    get_title_vector_e5("passage: " + str(text))
    for text in df["clean_job_title"].fillna("")
])

print("Candidates ready!")


# ---------- Ranking ----------

def search(query, favourites):
    if not query.strip():
        return pd.DataFrame(
            columns=["candidate_id", "job_title", "score"]
        )

    query_vector = get_title_vector_e5(
        "query: " + query.strip()
    )

    query_scores = cosine_similarity(
        query_vector.reshape(1, -1),
        X_e5,
    ).flatten()

    if favourites:
        starred_similarity = cosine_similarity(
            X_e5,
            X_e5[favourites],
        ).mean(axis=1)

        scores = (
            alpha * query_scores
            + (1 - alpha) * starred_similarity
        )
    else:
        scores = query_scores

    ranked = df[["job_title"]].copy()
    ranked["candidate_id"] = np.arange(len(df))
    ranked["score"] = scores

    return (
        ranked.sort_values("score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


def update_results(query, favourites):
    ranked = search(query, favourites)

    updates = []
    count = len(favourites)

    status = (
        f"★ {count} favourite{'s' if count != 1 else ''} selected"
        if count
        else "No favourites selected yet."
    )

    for position in range(10):
        if position < len(ranked):
            candidate = ranked.iloc[position]
            candidate_index = int(candidate["candidate_id"])
            is_favourite = candidate_index in favourites

            updates.extend([
                gr.update(
                    value=(
                        f"{position + 1}. "
                        f"{candidate['job_title']}\n"
                        f"Similarity score: {candidate['score']:.3f}"
                    ),
                    visible=True,
                ),
                gr.update(
                    value="★" if is_favourite else "☆",
                    visible=True,
                ),
                candidate_index,
            ])
        else:
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(visible=False),
                None,
            ])

    return [status] + updates


def new_search(query):
    favourites = []
    return [favourites] + update_results(query, favourites)


def toggle_favourite(query, candidate_index, favourites):
    favourites = list(favourites)

    if candidate_index is not None:
        candidate_index = int(candidate_index)

        if candidate_index in favourites:
            favourites.remove(candidate_index)
        else:
            favourites.append(candidate_index)

    return [favourites] + update_results(query, favourites)


def clear_favourites(query):
    favourites = []
    return [favourites] + update_results(query, favourites)


# ---------- Styling ----------

css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}

#hero {
    padding: 30px;
    margin-bottom: 22px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172554, #4338ca);
}

#hero h1, #hero p {
    color: white !important;
}

#hero h1 {
    margin: 0 0 10px;
    font-size: 32px;
}

#hero p {
    margin: 0;
    font-size: 16px;
    line-height: 1.6;
}

.candidate-row {
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    padding: 12px !important;
    margin-bottom: 10px !important;
}

.star-button {
    font-size: 28px !important;
    color: #d97706 !important;
    border-radius: 12px !important;
}

#search-button {
    border-radius: 12px;
    font-weight: 600;
}
"""


# ---------- Interface ----------

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>HR Candidate Finder</h1>
            <p>Find relevant candidates and refine your matches
            by starring the profiles you like.</p>
        </div>
        """
    )

    with gr.Row():
        query = gr.Textbox(
            label="Who are you looking for?",
            placeholder="e.g. aspiring human resources",
            scale=5,
        )

        search_button = gr.Button(
            "Search candidates",
            variant="primary",
            elem_id="search-button",
            scale=1,
        )

    gr.Markdown(
        "### Your top matches\n"
        "Click **☆** to favourite a candidate and update the ranking. "
        "Click **★** to remove them."
    )

    with gr.Row():
        status = gr.Markdown("No favourites selected yet.")
        clear_button = gr.Button("Clear favourites", size="sm")

    favourites = gr.State([])
    row_outputs = []
    star_buttons = []

    for _ in range(10):
        with gr.Row(elem_classes=["candidate-row"]):
            candidate_text = gr.Textbox(
                show_label=False,
                interactive=False,
                visible=False,
                scale=9,
                lines=2,
                container=False,
            )

            star_button = gr.Button(
                "☆",
                visible=False,
                scale=1,
                min_width=60,
                elem_classes=["star-button"],
            )

        candidate_index = gr.State(None)

        row_outputs.extend([
            candidate_text,
            star_button,
            candidate_index,
        ])

        star_buttons.append((star_button, candidate_index))

    all_outputs = [favourites, status] + row_outputs

    search_button.click(
        fn=new_search,
        inputs=query,
        outputs=all_outputs,
    )

    query.submit(
        fn=new_search,
        inputs=query,
        outputs=all_outputs,
    )

    clear_button.click(
        fn=clear_favourites,
        inputs=query,
        outputs=all_outputs,
    )

    for star_button, candidate_index in star_buttons:
        star_button.click(
            fn=toggle_favourite,
            inputs=[query, candidate_index, favourites],
            outputs=all_outputs,
        )

demo.launch(css=css)








