# HamroAI

HamroAI is a Nepali-focused language model project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darksunnp/HamroAI/blob/main/notebooks/hamroai_colab_inference.ipynb)

This repository provides the training and experimentation workspace, while the released artifacts are available on Hugging Face:

- Model: https://huggingface.co/darksunnp/hamroai-nepali-lora-v1
- Dataset: https://huggingface.co/datasets/darksunnp/hamroai

## Wrapper Website

A local web wrapper is included in the project at:

web_wrapper

It calls your Hugging Face Space API and provides a clean browser UI for prompt testing.

Quick run:

```bash
cd web_wrapper
pip install -r requirements.txt
python app.py
```

Open:

http://127.0.0.1:7861

## Quick Start

### 1) Install dependencies

Use Python 3.10+.

```bash
pip install -U torch transformers peft accelerate
```

If you want 4-bit or 8-bit quantized loading on NVIDIA GPUs, also install:

```bash
pip install -U bitsandbytes
```

### 2) Load the model from Hugging Face and run inference

Use this minimal script to load the adapter model and generate text.

```python
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

model_id = "darksunnp/hamroai-nepali-lora-v1"

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
model.to(device)
model.eval()

prompt = "Nepal ko rajdhani ke ho?"
inputs = tokenizer(
	prompt,
	return_tensors="pt",
	return_token_type_ids=False,
).to(device)
inputs.pop("token_type_ids", None)

with torch.no_grad():
	outputs = model.generate(
		**inputs,
		max_new_tokens=64,
		do_sample=False,
		repetition_penalty=1.1,
		no_repeat_ngram_size=3,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.pad_token_id,
	)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
