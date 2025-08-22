---
title: Multi-label News Classifier (Pro)
emoji: 📰
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 📰 Multi-label News Classifier (Pro)

This project is a **multi-label text classification app** that predicts the categories of news snippets.  
It was trained on a subset of the **20 Newsgroups dataset** and can assign multiple labels to a single text.

🚀 **Live Demo on Hugging Face Spaces**: [👉 Try it here](https://huggingface.co/spaces/rabiaceylan/news-classifier)

---

## 📂 Project Structure
- `app.py` → Gradio app for deployment  
- `model_multi.pkl` → Trained classifier (One-vs-Rest Logistic Regression)  
- `tfidf.pkl` → TF-IDF vectorizer  
- `requirements.txt` → Python dependencies  
- `README.md` → Project documentation  

---

## ⚙️ Features
- Multi-label classification with probabilities  
- Adjustable decision threshold  
- Interactive Gradio UI with plots and tables  
- Preloaded test examples for quick try-out  

---

## 🧠 Model
- **Algorithm**: One-vs-Rest Logistic Regression  
- **Features**: TF-IDF with 5000 max features  
- **Dataset**: Subset of 20 Newsgroups (`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.mideast`)  

### Performance (Multi-label setup)
- **Micro-F1**: ~0.88  
- **Macro-F1**: ~0.89  
- **Sample-F1**: ~0.85  

---

## 📊 Example Predictions
Input:  
NASA announces a new lunar mission with satellite support
Output:  
- **Predicted Labels**: `sci.space, talk.politics.mideast`  
- Probabilities chart displayed in UI  

---

## 🚀 Run Locally
```bash
# Clone this repo
git clone https://github.com/<your-username>/news-classifier.git
cd news-classifier

# Create virtual env & install dependencies
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the Gradio app
python app.py
