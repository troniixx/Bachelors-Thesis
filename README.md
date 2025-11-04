
# 📘 Bachelors Thesis - Explainable Phishing Detection

## 🧠 Overview

This project implements and evaluates Explainable Artificial Intelligence (XAI) techniques for phishing and spam email detection.

It combines classical machine learning and transformer-based classifiers with fact-checking features, LIME/SHAP explanations, and an optional interactive Streamlit prototype that demonstrates local interpretability for individual emails.

The goal is to make phishing detection transparent, educational, and secure, explaining why a message was flagged and allowing users to provide corrective feedback to improve future models.


## 🏗️ Project Structure

```bash

Bachelors-Thesis/
│
├──  app/  # Streamlit-based interactive demo
│  └──  app.py
│
├──  src/  # Core source code
│  ├──  models/  # Training, model configs, evaluation scripts
│  ├──  explain/  # LIME/SHAP explainability modules
│  ├──  data/  # Dataset handling and preprocessing
│  └──  helpers/  # Helper functions
│  └──  scripts/  # Shell scripts
│
├──  data/  # Datasets (cleaned / unified CSVs)
│  ├──  spam_assassin_cleaned.csv
│  ├──  zenodo_phishing.csv
│  ├──  enron_phishing.csv
│  └──  ...
│
├──  models/  # Trained pipelines and checkpoints
│  └──  tmp_sanity/pipeline.joblib
│  └──  pipeline_logreg.joblib
│  └──  ...
│
├──  feedback/  # User feedback (created automatically)
│  └──  feedback.csv
│
├──  requirements.txt
├──  README.md
```

## ⚙️ Installation

1. Create a virtual environment (Python 3.10+ recommended).
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
2. Install dependencies.
	```bash
	pip install -r requirements.txt
	```
3. (Optional) If using the transformer-based model, ensure PyTorch and Transformers are installed with MPS or GPU support.


## 🧩 Running the Interactive App

1. Train or place your model in the models/ directory (for example, models/pipeline_logreg.joblib or models/transformer_distilroberta-base/).

2. Start the Streamlit interface.
	```bash
	streamlit run app/app.py
	```
3. Paste or simulate an email, adjust the threshold, and view predictions with LIME explanations and optional FactChecker analysis.

## 🧠 Methodological Summary

| **Component** | **Description** |
| :------------- | :-------------- |
| **Datasets** | SpamAssassin, Zenodo Phishing, Enron Email Corpus |
| **Features** | Text (TF-IDF or SBERT), sender domain, TLD severity, URL obfuscation, fact-checking signals |
| **Models** | Naive Bayes, Logistic Regression, Random Forest, SVM, SBERT + LR, DistilRoBERTa |
| **Explainability** | Local – LIME, Global – SHAP |
| **Evaluation** | Accuracy, F1-score, ROC-AUC, cross-validation, robustness test on Enron |
| **Prototype** | Streamlit UI for single-email analysis with interactive explanations and user feedback collection |

## 🧪 Example Usage

Running the interactive prototype:
	```
	streamlit run app/app.py
	```


## 🔒 Privacy and Ethics

This demo is designed for educational and research purposes only. All email samples are synthetic or sourced from public datasets.

When using the prototype, do not paste any real personal or sensitive emails. Feedback is stored locally on your device only.
## 📑 Citation

If you reference this work in your thesis or reports:

Mert Erol. Explainable Phishing Detection: Combining Machine Learning, Transformer Models and Fact-Checking for Transparent Cybersecurity. Bachelor's Thesis, Department of Computational Linguistics, University of Zurich, 2025

## 🛠️ Future Work

- Integrate fact checking API (e.g. company domain validation or WHOIS lookup)
- Extend to multilingual phishing detection
- Optional active learning loop where user feedback retrains the model