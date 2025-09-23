cd "/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis"
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
from src.models.datasets import load_csv, load_many
import pandas as pd

print("Baseline sample:")
print(load_csv(Path("data/baseline_spam-ham.csv")).head())

print("\nSpamAssassin sample:")
print(load_csv(Path("data/spam_assassin.csv")).head())

print("\nZenodo sample:")
print(load_csv(Path("data/zenodo.csv")).head())

print("\nMerged:")
df = load_many([Path("data/baseline_spam-ham.csv"),
                Path("data/spam_assassin.csv"),
                Path("data/zenodo.csv")])
print(df.sample(5))
print(df.label.value_counts())
PY

# --- Outputs ---

'
Baseline sample:
                                                text sender  label
0  ounce feather bowl hummingbird opec moment ala...             1
1  wulvob get your medircations online qnb ikud v...             1
2   computer connection from cnn com wednesday es...             0
3  university degree obtain a prosperous future m...             1
4  thanks for all your answers guys i know i shou...             0

SpamAssassin sample:
                                                text sender  label
0  From ilug-admin@linux.ie Mon Jul 29 11:28:02 2...             0
1  From gort44@excite.com Mon Jun 24 17:54:21 200...             1
2  From fork-admin@xent.com Mon Jul 29 11:39:57 2...             1
3  From dcm123@btamail.net.cn Mon Jun 24 17:49:23...             1
4  From ilug-admin@linux.ie Mon Aug 19 11:02:47 2...             0

Zenodo sample:
                                                text                                             sender  label
0  Never agree to be a loser\n\nBuck up, your tro...                   Young Esposito <Young@iworld.de>      1
1  Befriend Jenna Jameson\n\n\nUpgrade your sex a...                       Mok <ipline's1983@icable.ph>      1
2  CNN.com Daily Top 10\n\n>+=+=+=+=+=+=+=+=+=+=+...  Daily Top 10 <Karmandeep-opengevl@universalnet...      1
3  Re: svn commit: r619753 - in /spamassassin/tru...                 Michael Parker <ivqrnai@pobox.com>      0
4  SpecialPricesPharmMoreinfo\n\n\nWelcomeFastShi...  Gretchen Suggs <externalsep1@loanofficertool.com>      1

Merged:
                                                    text                               sender  label
102148  WorldwideCanadianSafeSecure ForCustomersPillsP...       Elma Haas <risky@flatfish.com>      1
123140  Re: [Python-3000] range() issues On Wed, Apr 3...  Guido van Rossum <hoauf@python.org>      0
121252  Re: Hello - script question // Hello, Can you ...             MINO <xjvoqdh@gmail.com>      0
99626   Re: [Python-3000] Using *a for packing in list...   Thomas Wouters <fgxflg@python.org>      0
73045   agi consulting will be conducting the team dev...                                           0
label
1    67648
0    60750
Name: count, dtype: int64
'