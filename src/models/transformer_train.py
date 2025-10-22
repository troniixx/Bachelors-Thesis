#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# fine-tunes a transformer model for spam/phishing detection
import argparse
import numpy as np
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .datasets import load_many

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--model_name', default='distilroberta-base')
    ap.add_argument('--out_dir', default='models/distilroberta')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max_len', type=int, default=256)
    
    return ap.parse_args()

def main():
    args = parse_args()
    df = load_many(args.datasets)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = HFDataset.from_pandas(df[['text', 'label']])
    ds = ds.train_test_split(test_size=0.1, seed=42)
    
    def tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=args.max_len)
    
    ds = ds.map(tok, batched=True)
    ds = ds.rename_column('label', 'labels')
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=metrics
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()