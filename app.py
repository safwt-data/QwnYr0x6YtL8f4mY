import gradio as gr
import pandas as pd
from pathlib import Path

# Load the CSV from the same folder as this script
df = pd.read_csv("../data/clean_hr_data.csv")

def search(query):
    matches = df["job_title"].fillna("").str.contains(
        query.strip(),
        case=False,
        regex=False,
    )
    return df.loc[matches]

demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Search job titles"),
    outputs=gr.Dataframe(label="Candidates"),
    title="HR Candidate Search",
)

demo.launch()

