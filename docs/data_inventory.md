# HamroAI Data Inventory

## Text Corpora (for Pre-training)

| Dataset | Location | Approx Size | License | Type | Notes |
|---|---|---|---|---|---|
| IRIISNEPAL | data/raw/IRIISNEPAL/ | ~10.1 GB | See LICENSE | News articles | 6.4M articles from 99 news sites |
| OSCAR Nepali | data/raw/OSCAR Corpus (Nepali subset)/ | ~3.8 GB | CC-BY | Web crawl | Deduplicated version: ne_dedup.txt |
| Wikipedia Nepali | data/raw/Wikipedia Nepali dump/ | ~200 MB | CC-BY-SA | Encyclopedia | Individual article files in train/ and valid/ |

## NER / Structured Data (for Fine-tuning & Eval)

| Dataset | Location | Approx Size | License | Type | Notes |
|---|---|---|---|---|---|
| EverestNER | data/raw/EverestNER/ | ~50 MB | Research | NER (BIO tags) | 8 entity types, train/test split provided |

## Speech Data (for Phase 8)

| Dataset | Location | Approx Size | License | Type | Notes |
|---|---|---|---|---|---|
| Mozilla Common Voice | data/raw/Mozilla Common Voice/ | ~5 GB | CC-0 | ASR | train/dev/test splits provided |
| OpenSLR-54 | data/raw/OPENSLR Nepali dataset/ | ~20 GB | Apache 2.0 | ASR | 16 shards (asr_nepali_0 through _f) |

## Tokenizer Resources

| Dataset | Location | Type | Notes |
|---|---|---|---|
| Nepali Unigrams | data/tokenizer/nepali_unigrams.txt | Word list | 200k+ unique words with frequency |
| Brihat Sabdakosh | data/tokenizer/nepali-brihat-sabdakosh-json-main/ | Dictionary | 122k dictionary entries |

## Datasets NOT YET Downloaded

| Dataset | Priority | Why Needed | Phase |
|---|---|---|---|
| CC-100 Nepali | Medium | Additional pre-training text | Phase 2 |
| FLORES-200 Nepali-English | **HIGH** | Translation evaluation benchmark | Phase 6 |
| Bactrian-X / Aya Dataset | **HIGH** | Instruction tuning | Phase 5 |
| KU EN-NE Parallel Corpus (1.8M pairs) | **HIGH** | Translation training | Phase 5/7 |
| Nepali Health Q&A | Medium | Domain fine-tuning | Phase 5 |
| NLUE Benchmark | **HIGH** | Evaluation | Phase 6 |