# HamroAI Wrapper Website

A lightweight Flask wrapper website for the Hugging Face Space API endpoint.

## Deploy on Render

This project includes a root-level `render.yaml` blueprint, so deployment is straightforward.

1. Push this repository to GitHub.
2. In Render, click New + and select Blueprint.
3. Connect your GitHub repo and select this project.
4. Render reads `render.yaml` and creates the web service automatically.
5. Set environment variables in Render dashboard if needed:

- HAMROAI_SPACE_ID (default: darksunnp/HamroAI)
- HAMROAI_API_NAME (default: /generate)
- HF_TOKEN (only if your Hugging Face Space is private)

6. Deploy and open your Render URL.

## Run locally

1. Open a terminal in this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional environment variables:

- HAMROAI_SPACE_ID (default: darksunnp/HamroAI)
- HAMROAI_API_NAME (default: /generate)
- HF_TOKEN (only needed for private Spaces)
- PORT (default: 7861)

4. Start the server:

```bash
python app.py
```

Production-style local run:

```bash
gunicorn app:app --bind 0.0.0.0:7861
```

5. Open:

http://127.0.0.1:7861
