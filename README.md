# 📰 Dynamic News Classification with Zero-Shot Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-Zero--Shot-green)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)

## 📌 Project Overview
This project is a **Dynamic News Classifier** that uses **Zero-Shot Learning (BART/DistilBART)**. Unlike traditional classifiers that are limited to pre-defined labels, this model can categorize text into ANY category defined by the user at runtime.

## 🧠 Why Zero-Shot Learning?
Traditional models require thousands of examples for each category. With **Zero-Shot Learning**, the model understands the semantic relationship between the text and the labels provided. You can perform the classification instantly without further training.

## 🚀 How to Run
1. Clone the repo: `git clone https://github.com/rbacyln/news-classifier.git`
2. Install dependencies: `pip install transformers torch gradio`
3. Run the notebook: `ai_news_classifier_zeroshot.ipynb`

## 🖥️ Interactive UI
The project includes a **Gradio interface** where you can:
- Input a news headline.
- Provide custom labels (comma-separated).
- See real-time confidence scores for each label.

---
*Developed by Rabia Ceylan as a modern NLP implementation project.*