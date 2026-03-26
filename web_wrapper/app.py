import os
from threading import Lock

from flask import Flask, jsonify, render_template, request
from gradio_client import Client

app = Flask(__name__)

SPACE_ID = os.environ.get("HAMROAI_SPACE_ID", "darksunnp/HamroAI")
API_NAME = os.environ.get("HAMROAI_API_NAME", "/generate")
HF_TOKEN = os.environ.get("HF_TOKEN")
PORT = int(os.environ.get("PORT", "7861"))

_client = None
_client_lock = Lock()


def get_client() -> Client:
    global _client
    with _client_lock:
        if _client is None:
            if HF_TOKEN:
                _client = Client(SPACE_ID, hf_token=HF_TOKEN)
            else:
                _client = Client(SPACE_ID)
    return _client


@app.get("/")
def home():
    return render_template(
        "index.html",
        space_id=SPACE_ID,
        api_name=API_NAME,
    )


@app.post("/api/generate")
def generate() -> tuple:
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    max_new_tokens = payload.get("max_new_tokens", 80)

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        max_new_tokens = int(max_new_tokens)
    except (TypeError, ValueError):
        return jsonify({"error": "max_new_tokens must be an integer."}), 400

    max_new_tokens = max(8, min(256, max_new_tokens))

    try:
        client = get_client()
        result = client.predict(
            prompt=prompt,
            max_new_tokens=float(max_new_tokens),
            api_name=API_NAME,
        )
    except Exception as exc:
        return jsonify({"error": f"Generation failed: {exc}"}), 500

    return jsonify({"output": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
