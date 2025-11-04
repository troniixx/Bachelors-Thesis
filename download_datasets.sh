#!/bin/bash
# download_data.sh
set -e  # stop if any command fails

echo "⬇️  Downloading datasets from Google Drive..."
# Replace the link below with your folder URL
gdown --folder "https://drive.google.com/drive/folders/1dAnc6ukPf0kC3rjqlSAoTipZnS7uSxHH?usp=sharing" -O data/

echo "All datasets downloaded to data/"