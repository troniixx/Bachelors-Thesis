
# 📘 Bachelors Thesis - Explainable Phishing Detection

## 📚 Table of Contents

- [📘 Bachelors Thesis - Explainable Phishing Detection](#-bachelors-thesis---explainable-phishing-detection)
	- [📚 Table of Contents](#-table-of-contents)
	- [🧠 Overview](#-overview)
	- [🏗️ Project Structure](#️-project-structure)
	- [⚙️ Installation](#️-installation)
	- [🧩 Running the Interactive App and Model Selection](#-running-the-interactive-app-and-model-selection)
	- [🧠 Methodological Summary](#-methodological-summary)
	- [🔧 Training your own models](#-training-your-own-models)
	- [📋 Fact Checker](#-fact-checker)
	- [🔒 Privacy and Ethics](#-privacy-and-ethics)
	- [📑 Citation](#-citation)
	- [🛠️ Future Work](#️-future-work)

## 🧠 Overview

This project explores how Explainable Artificial Intelligence (XAI) can improve phishing and spam email detection by combining high-performance machine learning models with transparent, user-friendly explanations.

It integrates:
- Classical models (e.g., Logistic Regression, Naive Bayes, SVMs)
- Transformer-based classifiers (DistilRoBERTa)
- Factual risk indicators (domain validity, URL obfuscation, brand mismatch)
- Local explanation methods (LIME)

An interactive Streamlit prototype demonstrates how single-email predictions can be explained through token-level highlights and factual cues, and allow users to submit corrective feedback.

The overarching goal is to create phishing detection systems that are not only accurate, but also transparent, educational and user-centered, helping users understand why a message was flagged and encouraging safer email behavior.


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
│  └──  helpers/  # Helper functions (This can be ignored since only the cleaned datasets have been pushed to this repo)
│  └──  scripts/  # Shell scripts
│
├──  data/  # Datasets (cleaned / unified CSVs)
│  ├──  spam_assassin_cleaned.csv
│  ├──  zenodo_phishing.csv
│  ├──  enron_phishing.csv
│  └──  ...
│
├──  models/  # Trained pipelines and checkpoints
│  └──  runs/
│  		└──  20251022-124353 # Models trained by me provided to test
│  		└──  dated_run # Folder with models trained by you!
│  			└──  model_file
│
├──  feedback/  # User feedback (created automatically)
│  └──  feedback.csv
│
├──  requirements.txt
├──  README.md
├──  LICENSE
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

3. (Optional) Enable transformer-based models
If you plan to use the DistilRoBERTa model (or any other transformer), make sure PyTorch and transformers are installed with MPS/GPU support on your system

4. Download datasets (if missing)
   	Run the dataset download script:
   	```bash
	chmod +x src/scripts/download_datasets.sh
	./src/scripts/download_datasets.sh
   	```
NOTE: Google Drive sometimes rate-limits downloads. If the script fails, simpy retry.
As a fallback, you can manually download the datasets using the link printed in the error message.


## 🧩 Running the Interactive App and Model Selection

1. Place or train a model and store it inside the models/ directory.
	Examples:
	- models/runs/YOUR_RUNID/tfidf_bernoulli_nb
	- models/runs/YOUR_RUNID/transformer_distilroberta-base
	- models/runs/20251022-124353/transformer_distilroberta-base (default)

2. Start the Streamlit interface:
```bash
streamlit run app/app.py
```

1. Use the interface:
   - Paste or simulate an email.
   - Adjust the prediction threshold using the sidebar slider.
   - View the model output, LIME explanations and optional FactChecker results.
  
2. Select a model in the sidebar:
	On the right-hand side of the prototype (Streamlit sidebar), choose the model directory to load.
	- Important: When selecting models, choose the entire folder path, not individual files inside it.

## 🧠 Methodological Summary

| **Component** | **Description** |
| :------------- | :-------------- |
| **Datasets** | SpamAssassin, Zenodo Phishing, Enron Email Corpus |
| **Features** | Text (TF-IDF or SBERT), sender domain, TLD severity, URL obfuscation, fact-checking signals |
| **Models** | Naive Bayes, Logistic Regression, Random Forest, SVM, SBERT + LR, DistilRoBERTa |
| **Explainability** | Local – LIME, Global – SHAP |
| **Evaluation** | Accuracy, F1-score, ROC-AUC, cross-validation, robustness test on Enron |
| **Prototype** | Streamlit UI for single-email analysis with interactive explanations and user feedback collection |

## 🔧 Training your own models

If the data/ directory is empty, download the datasets first:
```bash
chmod +x src/scripts/download_datasets.sh
./src/scripts/download_datasets.sh
```
Note: Google Drive rate limits can occasionally cause failures.
If the script errors, simply run it again. As a fallback, manually download the files using the link shown in the error message.

1. Choose or customize a model

	All baseline models are defined in ```src/models/baselines.py ```
	You may:
	- Use the models already provided, or
	- customize/extend them (ensure you keep the same return format so the pipeline remains compatible).

2. Adjust Configurations (optional)

	Global settungs such as:
	- number of cross-validation folds
	- TF-IDF parameters
	- output directories
	- model hyperparameters
	can be changed in: ```src/models/config.py```

3. Run the full training pipeline

	Once your models and configurations are ready, start the training process:
	```bash
	chmod +x src/scripts/run_pipeline.sh
	./src/scripts/run_pipeline.sh
	```
	⚠️ Warning:
	This process may take a long time, especially when training SBERT or transformer-based models.

4. Where outputs are saved

	After the pipeline completes:
	- Models for the prototype are saved under ```models/runs/YOUR_RUN/MODEL_NAME/```(These folders can be selected in the app)
	- Predicitions on the Enron holdout corpus are saved under ```runs/YOUR_RUN/artifacts/preds/```
  
	These artifacts support:
	- quality inspection
	- error analysis
	- cross-domain robustness evaluation

## 📋 Fact Checker

The prototype includes a lightweight rule-based Fact Checker that highlights suspicious elements such as risky domains, obfuscated URLs, and brand inconsistencies.

If you want to expand or refine these factual signals, you can modify or extend the resources located in:
```data/fact_checking```

These files allow you to:
- add or update lists of trusted and untrusted domains,
- include additional brand–domain mappings,
- define new URL-risk patterns,
- enrich rule sets used during explainability.

The Fact Checker will automatically incorporate any additions the next time the app or pipeline runs, making it easy to build a more detailed and security-aware rule-based layer.

## 🔒 Privacy and Ethics

This prototype is intended solely for research and educational use. All example emails included in the project are synthetic or drawn from publicly available datasets.

When interacting with the app, do not paste real personal, confidential, or sensitive emails, as the system is not designed for processing private data.
Any feedback you provide through the interface is stored locally on your machine only and is never transmitted elsewhere.

The project emphasizes transparency and user awareness, but it should not be used as a production-level security tool without appropriate privacy, compliance and data-handling safeguards.

## 📑 Citation

If you reference this work in your thesis or reports:

Mert Erol. Explainable Phishing Detection: Combining Machine Learning, Transformer Models and Fact-Checking for Transparent Cybersecurity. Bachelor's Thesis, Department of Computational Linguistics, University of Zurich, 2025

## 🛠️ Future Work

- Integrate external fact-checking APIs
Add live checks such as WHOIS lookups, brand-domain matching, or security-advisory verification to enhance factual explanations and detect impersonation attempts more reliably.
- Extend to multilingual and cross-platform phishing detection
Support emails in additional languages and evaluate performance on non-email communication channels (e.g., messaging apps, ticketing systems, or social platforms).
- Improve explanation methods
Explore more advanced XAI techniques such as Integrated Gradients, attention rollout, or transformer-optimized SHAP variants to provide more stable and faithful explanations.
- Human-in-the-loop learning
Turn user feedback from the prototype into a true active-learning loop. Corrections from users could be used to update the model over time and improve robustness against new phishing strategies.
- Strengthen robustness and adversarial resilience
Incorporate adversarial training, homoglyph normalization, or robust embedding models to handle obfuscated URLs, brand impersonation, or subtly rephrased phishing attempts.
- Deployment in real-world email environments
Integrate the system as an extension or plugin in an email client or security dashboard, enabling live phishing analysis and real-time explanations.
- User-study driven evaluation
Conduct controlled user studies to assess how explanations influence trust, understanding, and phishing-avoidance behavior, and iterate the interface based on usability findings.