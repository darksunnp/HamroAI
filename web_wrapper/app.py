import os
import time
from threading import Lock

from flask import Flask, jsonify, render_template, request
from gradio_client import Client

app = Flask(__name__)

SPACE_ID = os.environ.get("HAMROAI_SPACE_ID", "darksunnp/HamroAI")
_api_name = os.environ.get("HAMROAI_API_NAME", "/generate")
API_NAME = _api_name if _api_name.startswith("/") else "/generate"
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


def reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


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

    errors = []
    api_candidates = [API_NAME]
    if API_NAME != "/generate":
        api_candidates.append("/generate")

    for attempt in range(3):
        if attempt > 0:
            reset_client()
            time.sleep(attempt)

        for api_name in api_candidates:
            try:
                client = get_client()
                result = client.predict(
                    prompt=prompt,
                    max_new_tokens=float(max_new_tokens),
                    api_name=api_name,
                )
                return jsonify({"output": result})
            except Exception as exc:
                errors.append(f"attempt={attempt + 1}, api={api_name}, error={exc}")

    err_text = errors[-1] if errors else "Unknown generation error"
    hint = None

    if "Expecting value" in err_text:
        hint = (
            "Hugging Face Space returned a non-JSON response. "
            "Common causes: Space cold start, temporary Space runtime error, "
            "or invalid API name."
        )
    elif "401" in err_text or "403" in err_text:
        hint = "Authentication failed. If your Space is public, ensure HF_TOKEN is removed from Render env."
    elif "404" in err_text:
        hint = "Endpoint not found. Confirm HAMROAI_API_NAME is /generate."

    app.logger.error("Generation failed. Details: %s", " | ".join(errors))
    return jsonify({"error": f"Generation failed: {err_text}", "hint": hint}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
