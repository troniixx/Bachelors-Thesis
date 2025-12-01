#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

"""
Clean SpamAssassin CSV:
- Extract first email address from the 'text' field
- Put it into a new column called 'sender'
- Keep 'text' and 'target' as-is
"""

import re
import pandas as pd
import argparse

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

# Regex for email addresses
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+')

def extract_sender(text: str) -> str:
    """Extract first email address from text (if any)."""
    match = EMAIL_REGEX.search(text)
    if match:
        return match.group(0).lower()
    return ""

def clean_spamassassin(input_file: str, output_file: str) -> None:
    # Load your raw CSV
    df = pd.read_csv(input_file)

    # Extract sender
    df["sender"] = df["text"].apply(extract_sender)

    # Reorder columns (optional)
    df = df[["text", "sender", "target"]]

    # Save back to CSV
    df.to_csv(output_file, index=False)
    print(f"Cleaned data written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean SpamAssassin CSV dataset.")
    
    parser.add_argument("input", type=str, help="Path to input SpamAssassin CSV file.")
    parser.add_argument("output", type=str, help="Path to output cleaned CSV file.")
    
    args = parser.parse_args()
    clean_spamassassin(args.input, args.output)
