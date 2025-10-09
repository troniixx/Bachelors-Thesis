#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import tarfile
import pandas as pd
from email.parser import BytesParser
from email import policy
from html import unescape
import re
from typing import Iterable, Optional, Tuple

HTML_TAG_RE = re.compile(r'<[^>]+>')
WS_RE = re.compile(r'[ \t\f\v]+')

def clean_ws(t: str) -> str:
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    t = WS_RE.sub(' ', t)
    t = re.sub(r'\n{3, }', '\n\n', t)
    return t.strip()

def html_to_text(h: str) -> str:
    return clean_ws(HTML_TAG_RE.sub(' ', unescape(h or '')))

def parse_email(raw: bytes) -> Tuple[str, str, str, str]:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return '', '', '', ''
    frm = (msg.get('From') or '').strip()
    to = (msg.get('To') or '').strip()
    subject = (msg.get('Subject') or '').strip()
    
    body = ''
    if msg.is_multipart():
        plain = None
        html = None
        
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain' and plain is None:
                try: plain = part.get_content().strip()
                except Exception: pass
            elif ctype == 'text/html' and html is None:
                try: html = part.get_content().strip()
                except Exception: pass
        if plain: body = plain
        elif html: body = html_to_text(html)
    else:
        try:
            payload = msg.get_content()
            if msg.get_content_type() == 'text/plain':
                body = html_to_text(payload or "")
            else:
                body = payload or ""
        except Exception:
            body = ''
            
    return frm, to, subject, clean_ws(body or "")

def load_enron_tar(tar_path: str, include_footers: Optional[Iterable[str]] = ("inbox","sent","sent_items"), exclude_footers: Optional[Iterable[str]] = ("deleted_items",), limit: Optional[int] = None, ) -> pd.DataFrame:
    from tqdm import tqdm
    
    inc = set(s.lower() for s in include_footers or [])
    exc = set(s.lower() for s in exclude_footers or [])
    rows = []
    n = 0
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get all members for progress tracking
        members = tar.getmembers()
        print(f"Processing {len(members)} files from tar archive...")
        
        for m in tqdm(members, desc="Processing emails"):
            if not m.isfile(): continue
            
            name = m.name
            if not name.startswith('maildir/'): continue
            lname = name.lower()
            
            if any(e in lname for e in exc): continue
            if inc and not any(i in lname for i in inc): continue
            
            try:
                f = tar.extractfile(m)
                if f is None: continue
                raw = f.read()
            except Exception:
                continue
            
            frm, to, sub, txt = parse_email(raw)
            rows.append({
                "from": frm, "to": to, "text": sub + txt, "label": 0, "path": name, "source": "enron"
            })
            n += 1
            
            # Update progress bar description with email count
            if n % 1000 == 0:
                tqdm.write(f"Processed {n} emails so far...")
            
            if limit and n >= limit: break
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    
    df = load_enron_tar("/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis/data/enron_mail.tar.gz")
    print(df.head())
    df.to_csv("/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis/data/enron_emails.csv", index=False)
    print(f"Total emails loaded: {len(df)}")