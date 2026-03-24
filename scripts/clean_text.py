"""
HamroAI Text Cleaning Pipeline
Processes raw Nepali text through sequential cleaning steps.
Outputs clean JSONL files to data/cleaned/
"""

import os
import re
import json
import unicodedata
import hashlib
from pathlib import Path

import pandas as pd

# === CONFIGURATION ===
RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# Nepali Devanagari Unicode range
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')

# PII patterns
PHONE_PATTERN = re.compile(r'\b(?:\+977|977|0)?[- ]?(?:98|97|96|01|061|021)\d{6,8}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


def normalize_unicode(text: str) -> str:
    """Step 1: Normalize to NFC (critical for Devanagari)."""
    return unicodedata.normalize('NFC', text)


def remove_html(text: str) -> str:
    """Step 2: Strip HTML tags and boilerplate."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities
    return text


def is_nepali(text: str, threshold: float = 0.3) -> bool:
    """Step 3: Simple heuristic — check ratio of Devanagari characters.
    For production, use fasttext lid.176.bin instead.
    """
    if len(text.strip()) == 0:
        return False
    devanagari_chars = len(DEVANAGARI_RANGE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return (devanagari_chars / total_alpha) >= threshold


def quality_filter(text: str) -> bool:
    """Step 5: Basic quality checks."""
    if len(text.strip()) < 50:
        return False
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:  # Too repetitive
            return False
    # Check if mostly numbers/punctuation
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < 0.3:
        return False
    return True


def scrub_pii(text: str) -> str:
    """Step 6: Remove phone numbers and emails."""
    text = PHONE_PATTERN.sub('[PHONE]', text)
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)      # Collapse spaces
    return text.strip()


def compute_hash(text: str) -> str:
    """For exact deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def clean_document(text: str, source: str) -> dict | None:
    """Run full pipeline on a single document."""
    text = normalize_unicode(text)
    text = remove_html(text)
    text = clean_whitespace(text)

    if not is_nepali(text):
        return None
    if not quality_filter(text):
        return None

    text = scrub_pii(text)

    return {
        "text": text,
        "source": source,
        "hash": compute_hash(text),
    }


def process_oscar():
    """Process OSCAR corpus."""
    print("Processing OSCAR corpus...")
    seen_hashes = set()
    output_file = CLEANED_DIR / "oscar_cleaned.jsonl"
    count = 0

    dedup_path = RAW_DIR / "OSCAR Corpus (Nepali subset)" / "ne_dedup.txt"
    if not dedup_path.exists():
        print(f"  OSCAR dedup file not found at {dedup_path}")
        return

    with open(output_file, 'w', encoding='utf-8') as out:
        with open(dedup_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = clean_document(line.strip(), source="oscar")
                if doc and doc["hash"] not in seen_hashes:
                    seen_hashes.add(doc["hash"])
                    out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    count += 1

    print(f"  OSCAR: {count} documents saved")


def process_wikipedia():
    """Process Wikipedia Nepali dump."""
    print("Processing Wikipedia...")
    seen_hashes = set()
    output_file = CLEANED_DIR / "wikipedia_cleaned.jsonl"
    count = 0

    wiki_dir = RAW_DIR / "Wikipedia Nepali dump" / "train" / "train"
    if not wiki_dir.exists():
        print(f"  Wikipedia directory not found at {wiki_dir}")
        return

    with open(output_file, 'w', encoding='utf-8') as out:
        for txt_file in sorted(wiki_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding='utf-8')
                doc = clean_document(text, source="wikipedia")
                if doc and doc["hash"] not in seen_hashes:
                    seen_hashes.add(doc["hash"])
                    out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    count += 1
            except Exception as e:
                print(f"  Error reading {txt_file}: {e}")

    print(f"  Wikipedia: {count} documents saved")


def process_iriisnepal():
    """Process IRIISNEPAL corpus.
    Data is stored as .parquet files with columns: [index, Article, Source]
    - 46 train shards + 11 test shards = 57 parquet files
    - ~6.4M articles total from 99 Nepali news websites
    """
    print("Processing IRIISNEPAL...")
    seen_hashes = set()
    output_file = CLEANED_DIR / "iriisnepal_cleaned.jsonl"
    count = 0
    skipped = 0

    data_dir = RAW_DIR / "IRIISNEPAL" / "data"
    if not data_dir.exists():
        print(f"  IRIISNEPAL data directory not found at {data_dir}")
        print("  Check if Git LFS files need to be pulled: cd data/raw/IRIISNEPAL && git lfs pull")
        return

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("  No .parquet files found. Git LFS files may not be pulled.")
        print("  Run: cd data/raw/IRIISNEPAL && git lfs pull")
        return

    print(f"  Found {len(parquet_files)} parquet files")

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, filepath in enumerate(parquet_files):
            try:
                df = pd.read_parquet(filepath)
                file_count = 0

                for _, row in df.iterrows():
                    text = str(row.get('Article', ''))
                    source_site = str(row.get('Source', 'iriisnepal'))

                    doc = clean_document(text, source=source_site)
                    if doc and doc["hash"] not in seen_hashes:
                        seen_hashes.add(doc["hash"])
                        out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        count += 1
                        file_count += 1
                    else:
                        skipped += 1

                print(f"  [{i+1}/{len(parquet_files)}] {filepath.name}: {file_count} docs kept, {df.shape[0] - file_count} filtered")

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")

    print(f"  IRIISNEPAL: {count} documents saved, {skipped} filtered/duplicates removed")


if __name__ == "__main__":
    print("=" * 60)
    print("HamroAI Text Cleaning Pipeline")
    print("=" * 60)

    process_oscar()
    process_wikipedia()
    process_iriisnepal()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of cleaned data:")
    total_size = 0
    for f in CLEANED_DIR.glob("*.jsonl"):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  TOTAL: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print("=" * 60)