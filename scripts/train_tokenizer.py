# scripts/train_tokenizer.py
"""
Phase 3: Train a custom BPE tokenizer for Nepali.
Uses HuggingFace tokenizers library (fast Rust-backed implementation).
"""

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DEDUPED_FILE = Path("data/cleaned/deduped/all_cleaned_deduped.jsonl")
OUTPUT_DIR = Path("data/tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 32_000


def text_iterator():
    """Stream text from deduplicated corpus — never loads everything into RAM."""
    with open(DEDUPED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            yield doc["text"]


def train_tokenizer():
    print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE}")
    print(f"Reading from: {DEDUPED_FILE}")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: split on whitespace + punctuation
    # IMPORTANT: Do NOT use ByteLevel — it converts Devanagari characters to
    # 3 bytes each (UTF-8), wasting most of the 32K vocab budget on byte merges.
    # Whitespace splits on spaces/punctuation and feeds whole Unicode characters
    # to BPE, so it can learn proper Nepali subwords directly.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Decoder: handles re-joining BPE subword pieces back into text
    tokenizer.decoder = decoders.BPEDecoder()

    # No byte-level post-processor needed

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )

    # Train on streaming text
    print("Training... (this will take 1-3 hours on 23 GB)")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Save
    out_path = OUTPUT_DIR / "nepali_bpe_32k.json"
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def measure_fertility(tokenizer, sample_size=10000):
    """Measure tokens-per-word (fertility). Target: 1.2–1.5 for Nepali."""
    print(f"\nMeasuring fertility on {sample_size} documents...")

    total_tokens = 0
    total_words = 0

    count = 0
    for text in text_iterator():
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded.ids)
        total_words += len(text.split())
        count += 1
        if count >= sample_size:
            break

    fertility = total_tokens / total_words
    print(f"  Total words:  {total_words:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Fertility:    {fertility:.3f} tokens/word")

    if fertility <= 1.5:
        print("  ✓ EXCELLENT — within target range (1.2–1.5)")
    elif fertility <= 2.0:
        print("  ~ GOOD — slightly above target but acceptable")
    else:
        print("  ✗ HIGH — consider increasing vocab size or retraining")

    return fertility


def compare_with_others():
    """Compare against other tokenizers to show the advantage of a custom Nepali one."""
    from transformers import AutoTokenizer

    test_texts = [
        "नेपालको राजधानी काठमाडौं हो।",
        "प्रधानमन्त्री कार्यालयले आज एक विज्ञप्ति जारी गर्दै राष्ट्रिय सुरक्षाको विषयमा छलफल गरेको जनाएको छ।",
        "मलाई यो new phone चाहिन्छ, Amazon बाट order गर्छु",  # Nepanglish
    ]

    custom_tok = Tokenizer.from_file(str(OUTPUT_DIR / "nepali_bpe_32k.json"))

    # Freely available tokenizers to compare against (no gated access needed)
    comparison_models = [
        ("GPT-2", "gpt2"),
        ("Mistral-7B", "mistralai/Mistral-7B-v0.1"),
    ]

    # Try adding Llama if the user has access
    try:
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        comparison_models.append(("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"))
    except Exception:
        pass  # No access, skip it

    print("\nComparing tokenizers on Nepali text:")
    print("=" * 70)

    for text in test_texts:
        custom_tokens = len(custom_tok.encode(text).ids)
        print(f"\n  Text: {text[:70]}")
        print(f"    {'Tokenizer':<20} {'Tokens':>8}  {'vs Custom':>10}")
        print(f"    {'-'*42}")
        print(f"    {'HamroAI (Custom)':<20} {custom_tokens:>8}  {'baseline':>10}")

        for name, model_id in comparison_models:
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                other_tokens = len(tok.encode(text))
                ratio = other_tokens / max(custom_tokens, 1)
                print(f"    {name:<20} {other_tokens:>8}  {ratio:>9.1f}x")
            except Exception as e:
                print(f"    {name:<20} {'skipped':>8}  {str(e)[:30]}")

    print("\n" + "=" * 70)
    print("  Lower token count = more efficient for Nepali text")
    print("  A ratio > 1.0x means your custom tokenizer wins by that factor")


if __name__ == "__main__":
    tokenizer = train_tokenizer()
    measure_fertility(tokenizer)
    compare_with_others()