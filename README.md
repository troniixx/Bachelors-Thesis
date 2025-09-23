# Bachelors Thesis - Phishing Detection with Explainable AI

## Overview

This project implements a phishing/spam detection pipeline that combines machine learning with a custom built fact checker and XAI methods. The system is designed to:

- Detect phishing and spam emails using classical ML models (Naive Bayes, Logistic Regression, Random Forest, etc.)
- TBD: Detect phishing and spam emails using a transformer / LLM based model.
- Enrich detection with fact checking features (brand mismatch, risky TLDs, obfuscated URLs, suspicious claims)
- Provide human readable explanations of classification decisicions (via SHAP/LIME + fact checker outputs)
- Evaluate models under normal conditions and adversarial attacks to study robustness and security tradeoffs

This thesis addresses the question:
How can XAI systems provide useful explanations for phishing detection without creating security vulnerbilities?

## Datasets

- Baseline Spam-Ham: simple labeled text dataset
- SpamAssassin: raw emails with headers in text, target lables
- Zenodo Dataset: richer structure (sender, subject, body, urls, label)
