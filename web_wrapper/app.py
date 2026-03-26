import os
import time
import traceback
import json
import uuid
import urllib.error
import urllib.request
from threading import Lock, Thread

from flask import Flask, jsonify, render_template, request
from gradio_client import Client

app = Flask(__name__)

SPACE_ID = os.environ.get("HAMROAI_SPACE_ID", "darksunnp/HamroAI")
_api_name = os.environ.get("HAMROAI_API_NAME", "/generate")
API_NAME = _api_name if _api_name.startswith("/") else "/generate"
HF_TOKEN = os.environ.get("HF_TOKEN")
PORT = int(os.environ.get("PORT", "7861"))


def build_space_base_url(space_id: str) -> str:
    parts = space_id.split("/", 1)
    if len(parts) != 2:
        return f"https://{space_id}.hf.space"
    owner, name = parts
    return f"https://{owner}-{name}.hf.space"


SPACE_BASE_URL = os.environ.get("HAMROAI_SPACE_BASE_URL", build_space_base_url(SPACE_ID))

_client = None
_client_lock = Lock()
_jobs = {}
_jobs_lock = Lock()


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


class GenerationError(Exception):
    def __init__(self, message: str, hint: str | None = None, details: str | None = None):
        super().__init__(message)
        self.hint = hint
        self.details = details


def call_space_run_endpoint(prompt: str, max_new_tokens: int, api_name: str) -> str:
    run_url = f"{SPACE_BASE_URL}/gradio_api/run{api_name}"
    payload = {
        # Gradio run endpoints expect positional args in `data`.
        "data": [prompt, float(max_new_tokens)],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        run_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Some Gradio Spaces only accept queued calls via /gradio_api/call/*.
        if exc.code == 404 and "join the queue" in body.lower():
            return call_space_queue_endpoint(prompt, max_new_tokens, api_name)
        raise

    parsed = json.loads(text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("output"), str):
            return parsed["output"]
        data = parsed.get("data")
        if isinstance(data, list) and data:
            return str(data[0])
    return str(parsed)


def _extract_output_from_payload(payload_obj):
    if isinstance(payload_obj, dict):
        if isinstance(payload_obj.get("output"), str):
            return payload_obj["output"]
        data = payload_obj.get("data")
        if isinstance(data, list) and data:
            return str(data[0])
    if isinstance(payload_obj, list) and payload_obj:
        return str(payload_obj[0])
    return str(payload_obj)


def _parse_sse_result(text: str):
    data_lines = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    # Parse from the end because the final SSE message typically contains the completed payload.
    for item in reversed(data_lines):
        if not item:
            continue
        try:
            parsed = json.loads(item)
            return _extract_output_from_payload(parsed)
        except Exception:
            continue

    raise ValueError("Could not parse queue response payload")


def call_space_queue_endpoint(prompt: str, max_new_tokens: int, api_name: str) -> str:
    call_url = f"{SPACE_BASE_URL}/gradio_api/call{api_name}"
    payload = {"data": [prompt, float(max_new_tokens)]}
    body = json.dumps(payload).encode("utf-8")

    start_req = urllib.request.Request(
        call_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(start_req, timeout=120) as resp:
        start_text = resp.read().decode("utf-8", errors="replace")

    start_payload = json.loads(start_text)
    event_id = start_payload.get("event_id")
    if not event_id:
        return _extract_output_from_payload(start_payload)

    result_url = f"{call_url}/{event_id}"
    result_req = urllib.request.Request(result_url, method="GET")
    with urllib.request.urlopen(result_req, timeout=180) as resp:
        result_text = resp.read().decode("utf-8", errors="replace")

    return _parse_sse_result(result_text)


def perform_generation(prompt: str, max_new_tokens: int) -> str:
    errors = []
    tracebacks = []
    api_candidates = [API_NAME]
    if API_NAME != "/generate":
        api_candidates.append("/generate")

    for attempt in range(3):
        if attempt > 0:
            reset_client()
            time.sleep(attempt)

        for api_name in api_candidates:
            try:
                return call_space_run_endpoint(prompt, max_new_tokens, api_name)
            except urllib.error.HTTPError as exc:
                try:
                    response_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    response_text = "<no response body>"
                errors.append(
                    f"attempt={attempt + 1}, mode=direct, api={api_name}, http={exc.code}, body={response_text[:300]}"
                )
                tracebacks.append(traceback.format_exc())
            except Exception as exc:
                errors.append(f"attempt={attempt + 1}, mode=direct, api={api_name}, error={exc}")
                tracebacks.append(traceback.format_exc())

            try:
                client = get_client()
                result = client.predict(
                    prompt=prompt,
                    max_new_tokens=float(max_new_tokens),
                    api_name=api_name,
                )
                return str(result)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1}, mode=client, api={api_name}, error={exc}")
                tracebacks.append(traceback.format_exc())

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

    details = " | ".join(errors)
    app.logger.error("Generation failed. Details: %s", details)
    for idx, tb in enumerate(tracebacks, start=1):
        app.logger.error("Generation traceback #%s:\n%s", idx, tb)

    raise GenerationError(err_text, hint=hint, details=details)


def run_generation_job(job_id: str, prompt: str, max_new_tokens: int) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        output = perform_generation(prompt, max_new_tokens)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["output"] = output
    except GenerationError as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"Generation failed: {exc}"
            _jobs[job_id]["hint"] = exc.hint
            _jobs[job_id]["details"] = exc.details
    except Exception as exc:
        tb = traceback.format_exc()
        app.logger.error("Unhandled generation job error: %s\n%s", exc, tb)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"Unhandled error: {exc}"
            _jobs[job_id]["details"] = tb


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

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": time.time(),
        }

    worker = Thread(target=run_generation_job, args=(job_id, prompt, max_new_tokens), daemon=True)
    worker.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/result/<job_id>")
def get_result(job_id: str) -> tuple:
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify(job), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
