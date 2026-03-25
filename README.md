# HamroAI

HamroAI is a Nepali-focused language model project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12K4MY2MRbOsszTrOyv6BUVp4IUwFYBBb?usp=sharing)

This repository provides the training and experimentation workspace, while the released artifacts are available on Hugging Face:

- Model: https://huggingface.co/darksunnp/hamroai-nepali-lora-v1
- Dataset: https://huggingface.co/datasets/darksunnp/hamroai

## Quick Start

### 1) Install dependencies

Use Python 3.10+.

```bash
%pip uninstall -y bitsandbytes
%pip install -q -U "transformers==4.45.2" "peft==0.12.0" "accelerate==0.34.2" "huggingface_hub>=0.24,<1.0"
```

### 2) Load the model from Hugging Face and run inference

Use this minimal script to load the adapter model and generate text.

```python
import os
import torch
import transformers
import peft
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Public repos do not require auth; this avoids noisy implicit-token lookups in Colab.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)

model_id = "darksunnp/hamroai-nepali-lora-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    print("Loading model in fp16 on GPU (bitsandbytes disabled for compatibility).")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    model.to("cuda")
    device = "cuda"
else:
    print("Loading model on CPU.")
    model = AutoPeftModelForCausalLM.from_pretrained(model_id)
    model.to("cpu")
    device = "cpu"

model.eval()
print(f"Loaded model on: {device}")
```


```python
def generate_text(prompt, max_new_tokens=80):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Nepal ko rajdhani ke ho?"
print(generate_text(prompt))
```

```Python
# Try your own prompt
user_prompt = "Write a short paragraph about Nepal in Nepali."
print(generate_text(user_prompt, max_new_tokens=120))
```

## Optional: Faster Inference with Quantization (GPU)

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

model_id = "darksunnp/hamroai-nepali-lora-v1"

bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.float16,
	bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(
	model_id,
	quantization_config=bnb_config,
	device_map="auto",
)
model.eval()

prompt = "Write two lines about Nepal in Nepali."
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
inputs.pop("token_type_ids", None)

with torch.no_grad():
	outputs = model.generate(
		**inputs,
		max_new_tokens=80,
		do_sample=True,
		temperature=0.7,
		top_p=0.9,
	)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Dataset

The training dataset is published at:

https://huggingface.co/datasets/darksunnp/hamroai

You can load it with:

```python
from datasets import load_dataset

ds = load_dataset("darksunnp/hamroai")
print(ds)
```

## Notes

- If you run into tokenizer argument errors during generation, keep this safeguard:
  - return_token_type_ids=False
  - inputs.pop("token_type_ids", None)
- For best quality, use the same tokenizer/model pairing from the same Hugging Face repo.

## License

Please follow the model and dataset card terms on Hugging Face.
