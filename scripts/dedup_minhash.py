"""
Cross-source fuzzy deduplication using MinHash + LSH.
Run AFTER clean_text.py

STREAMING VERSION — never loads all documents into memory.
Reads from input JSONL files line-by-line, writes kept documents
directly to the output file. Only the LSH index stays in RAM.
"""

import json
import gc
from pathlib import Path
from datasketch import MinHash, MinHashLSH

CLEANED_DIR = Path("data/cleaned")
DEDUPED_DIR = Path("data/cleaned/deduped")
DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.8  # Jaccard similarity threshold for near-duplicates
NUM_PERM = 64    # Reduced from 128 — halves RAM for LSH index, still effective


def get_minhash(text: str) -> MinHash:
    """Create MinHash from text using word-level shingles."""
    m = MinHash(num_perm=NUM_PERM)
    words = text.split()
    # Use 5-word shingles
    for i in range(len(words) - 4):
        shingle = ' '.join(words[i:i+5])
        m.update(shingle.encode('utf-8'))
    return m


def deduplicate():
    """
    Stream through cleaned JSONL files, deduplicate across sources,
    and write output — all without loading everything into memory.

    Memory usage: Only the LSH index is held in RAM (~2-4 GB for 7M docs).
    Documents are read one-by-one and written immediately if kept.
    """
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    output_file = DEDUPED_DIR / "all_cleaned_deduped.jsonl"

    kept_count = 0
    duplicate_count = 0
    doc_index = 0

    # Collect input files
    jsonl_files = sorted(CLEANED_DIR.glob("*.jsonl"))
    total_lines = 0
    for jf in jsonl_files:
        with open(jf, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    print(f"Total documents to process: {total_lines:,}")

    # Stream: read one doc at a time, check against LSH, write immediately if unique
    with open(output_file, 'w', encoding='utf-8') as out:
        for jsonl_file in jsonl_files:
            print(f"Processing {jsonl_file.name}...")
            file_kept = 0
            file_dup = 0

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc["text"]

                    # Skip very short texts (can't make meaningful shingles)
                    words = text.split()
                    if len(words) < 5:
                        doc_index += 1
                        duplicate_count += 1
                        file_dup += 1
                        continue

                    mh = get_minhash(text)

                    # Check if near-duplicate exists in index
                    result = lsh.query(mh)
                    if len(result) > 0:
                        duplicate_count += 1
                        file_dup += 1
                    else:
                        # Unique — insert into index and write to output
                        try:
                            lsh.insert(f"d{doc_index}", mh)
                            out.write(line)  # Write original line directly (already JSON)
                            kept_count += 1
                            file_kept += 1
                        except ValueError:
                            duplicate_count += 1
                            file_dup += 1

                    doc_index += 1
                    if doc_index % 100000 == 0:
                        print(f"  {doc_index:,}/{total_lines:,} — kept: {kept_count:,}, dupes: {duplicate_count:,}")

            print(f"  {jsonl_file.name}: kept {file_kept:,}, removed {file_dup:,}")

            # Free memory after each file
            gc.collect()

    print()
    print(f"DONE — Kept: {kept_count:,}, Removed: {duplicate_count:,}")
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Final corpus: {output_file} ({size_mb:.1f} MB / {size_mb/1024:.2f} GB)")


if __name__ == "__main__":
    deduplicate()