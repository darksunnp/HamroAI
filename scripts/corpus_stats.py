"""
Generate statistics for cleaned corpus.
Run AFTER dedup_minhash.py
"""

import json
from pathlib import Path
from collections import Counter

DEDUPED_FILE = Path("data/cleaned/deduped/all_cleaned_deduped.jsonl")


def compute_stats():
    total_docs = 0
    total_chars = 0
    total_words = 0
    source_counts = Counter()
    doc_lengths = []

    with open(DEDUPED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            source = doc.get("source", "unknown")

            total_docs += 1
            total_chars += len(text)
            words = len(text.split())
            total_words += words
            doc_lengths.append(words)
            source_counts[source] += 1

    doc_lengths.sort()
    median_idx = len(doc_lengths) // 2

    print("=" * 60)
    print("HamroAI Corpus Statistics")
    print("=" * 60)
    print(f"Total documents:     {total_docs:,}")
    print(f"Total characters:    {total_chars:,}")
    print(f"Total words:         {total_words:,}")
    print(f"Avg words/doc:       {total_words/max(total_docs,1):.0f}")
    print(f"Median words/doc:    {doc_lengths[median_idx]:,}")
    print(f"Approx size:         {total_chars / (1024**3):.2f} GB")
    print()
    print("Documents by source:")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    compute_stats()