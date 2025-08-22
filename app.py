# app.py — Multi-label News Classifier (Gradio)
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; fine locally too
import matplotlib.pyplot as plt
import gradio as gr

# --- Load artifacts (files must be in the same folder) ---
MODEL_PATH = os.getenv("MODEL_PATH", "model_multi.pkl")
VECT_PATH  = os.getenv("VECT_PATH",  "tfidf.pkl")

ovr = joblib.load(MODEL_PATH)        # OneVsRest(LogisticRegression)
vectorizer = joblib.load(VECT_PATH)  # TfidfVectorizer

# Label order used during training
label_names = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast']

def predict_multilabel_pro(text: str, threshold: float = 0.50):
    """
    Returns:
      - a text summary with active labels
      - a pandas DataFrame of sorted probabilities
      - a matplotlib Figure (bar chart)
    """
    if not text or not text.strip():
        df_empty = pd.DataFrame({"label": label_names, "probability": [0.0]*len(label_names)})
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(df_empty["label"], df_empty["probability"])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Class probabilities")
        fig.tight_layout()
        return "Please enter some text.", df_empty, fig

    v = vectorizer.transform([text])
    proba = ovr.predict_proba(v)[0]  # shape: (n_labels,)

    df = pd.DataFrame({"label": label_names, "probability": proba})
    df_sorted = df.sort_values("probability", ascending=False).reset_index(drop=True)

    active = df_sorted[df_sorted["probability"] >= threshold]["label"].tolist()
    if not active and len(df_sorted) > 0:
        active = [df_sorted.iloc[0]["label"]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(df_sorted["label"], df_sorted["probability"])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title("Class probabilities")
    fig.tight_layout()

    pred_text = f"Predicted labels (thr={threshold:.2f}): " + ", ".join(active)
    return pred_text, df_sorted, fig

# --- Gradio UI ---
demo = gr.Interface(
    fn=predict_multilabel_pro,
    inputs=[
        gr.Textbox(label="News text", lines=6, placeholder="Paste a news snippet..."),
        gr.Slider(0.10, 0.90, value=0.50, step=0.05, label="Threshold"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Labels"),
        # headers shown when type="pandas"
        gr.Dataframe(label="Sorted probabilities", type="pandas"),
        gr.Plot(label="Probability chart"),
    ],
    title="Multi-label News Classifier (Pro)",
    description=(
        "Enter a news snippet. The model predicts multiple categories with probabilities. "
        "Adjust the threshold to control which labels become active."
    ),
    examples=[
        ["NASA announces a new lunar mission with satellite support", 0.5],
        ["The baseball team celebrated their championship win yesterday", 0.5],
        ["Advances in computer graphics make video games more realistic", 0.5],
        ["Political leaders met in the Middle East to discuss peace", 0.5],
        ["Scientists discovered a new black hole in deep space", 0.5],
        ["Fans are excited about the upcoming baseball season", 0.5],
        ["3D rendering and visualization are core topics in computer graphics", 0.5],
    ],
)

# Keep queueing (nice for concurrency); locally launch when run directly
demo = demo.queue()

if __name__ == "__main__":
    demo.launch()
