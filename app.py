import gradio as gr
import pandas as pd

# Path relative to app.py
data_path = "data/clean_hr_data.csv"
df = pd.read_csv(data_path)

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

demo.launch(share=True)



