"""
Compare HamroAI custom Nepali tokenizer against state-of-the-art model tokenizers.
Includes GPT-4o/5 (via tiktoken), Gemma 2, Qwen 2.5, Mistral, GPT-2.
"""

import tiktoken
from tokenizers import Tokenizer
from pathlib import Path
from transformers import AutoTokenizer

OUTPUT_DIR = Path("data/tokenizer")
custom_tok = Tokenizer.from_file(str(OUTPUT_DIR / "nepali_bpe_32k.json"))

test_texts = [
    ("Short", "नेपालको राजधानी काठमाडौं हो।"),
    ("Long", "प्रधानमन्त्री कार्यालयले आज एक विज्ञप्ति जारी गर्दै राष्ट्रिय सुरक्षाको विषयमा छलफल गरेको जनाएको छ।"),
    ("Nepanglish", "मलाई यो new phone चाहिन्छ, Amazon बाट order गर्छु"),
    ("News", "काठमाडौं । पश्चिम नेपालमा पछिल्लो समय आगलागीका घटना बढ्न थालेको छ। लुम्बिनी, कर्णाली र सुदूरपश्चिम प्रदेशमा आगलागीका घटना बर्सेनि बढ्दै गएका छन्।"),
    ("English", "The quick brown fox jumps over the lazy dog."),
]

# GPT-4o / GPT-5 use o200k_base encoding
gpt_tok = tiktoken.get_encoding("o200k_base")

# Open-weight model tokenizers
hf_models = [
    ("Gemma-2-9B (Google)", "google/gemma-2-9b"),
    ("Qwen-2.5-7B (Alibaba)", "Qwen/Qwen2.5-7B"),
    ("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"),
    ("GPT-2 (OpenAI)", "gpt2"),
]

# Load HF tokenizers
loaded = {}
for name, mid in hf_models:
    try:
        loaded[name] = AutoTokenizer.from_pretrained(mid)
    except Exception as e:
        print(f"  Could not load {name}: {e}")

print("=" * 80)
print("  HamroAI Custom Tokenizer vs State-of-the-Art Models (Nepali Text)")
print("=" * 80)

for label, text in test_texts:
    ct = len(custom_tok.encode(text).ids)
    gt = len(gpt_tok.encode(text))

    display = text[:70] + ("..." if len(text) > 70 else "")
    print(f"\n  [{label}] {display}")
    header = "Tokenizer"
    print(f"    {header:<28} {'Tokens':>7}  {'vs HamroAI':>10}")
    print(f"    {'-' * 48}")
    print(f"    {'>> HamroAI (Yours) <<':<28} {ct:>7}  {'baseline':>10}")
    print(f"    {'GPT-4o / GPT-5 (OpenAI)':<28} {gt:>7}  {gt/ct:>9.1f}x")

    for name, tok in loaded.items():
        ot = len(tok.encode(text))
        print(f"    {name:<28} {ot:>7}  {ot/ct:>9.1f}x")

print()
print("=" * 80)
print("  Ratio > 1.0x = HamroAI uses fewer tokens (wins)")
print("  Ratio < 1.0x = Other model uses fewer tokens (loses)")
print()
print("  NOTE: GPT-4o/5 use the same 'o200k_base' tokenizer (200K vocab).")
print("  Claude & Gemini tokenizers are not publicly available,")
print("  but Gemma-2 is Google's closest open proxy for Gemini's tokenizer.")
print("=" * 80)
