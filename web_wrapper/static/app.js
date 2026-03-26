const form = document.getElementById("generate-form");
const promptEl = document.getElementById("prompt");
const tokenEl = document.getElementById("max_new_tokens");
const tokenOutputEl = document.getElementById("token-output");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status-pill");
const submitBtn = document.getElementById("submit-btn");

tokenEl.addEventListener("input", () => {
  tokenOutputEl.textContent = tokenEl.value;
});

function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseApiError(data) {
  let message = data.error || "Unknown server error";
  if (data.hint) {
    message += `\n\nHint: ${data.hint}`;
  }
  if (data.details) {
    message += `\n\nDetails: ${data.details}`;
  }
  return message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus("error", "Missing prompt");
    outputEl.textContent = "Please enter a prompt.";
    return;
  }

  submitBtn.disabled = true;
  setStatus("loading", "Generating...");
  outputEl.textContent = "Submitting generation job...";

  try {
    const startResponse = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        max_new_tokens: Number(tokenEl.value),
      }),
    });

    const startRawText = await startResponse.text();
    let startData = null;
    try {
      startData = startRawText ? JSON.parse(startRawText) : {};
    } catch {
      startData = { error: "Server returned a non-JSON response.", details: startRawText.slice(0, 500) };
    }

    if (!startResponse.ok) {
      throw new Error(parseApiError(startData));
    }

    const jobId = startData.job_id;
    if (!jobId) {
      throw new Error("No job_id returned from server.");
    }

    setStatus("loading", "Queued...");
    outputEl.textContent = `Job queued: ${jobId}\nWaiting for result...`;

    const startedAt = Date.now();
    const maxWaitMs = 8 * 60 * 1000;

    while (true) {
      await sleep(1500);

      const pollResponse = await fetch(`/api/result/${jobId}`);
      const pollRawText = await pollResponse.text();
      let pollData = null;
      try {
        pollData = pollRawText ? JSON.parse(pollRawText) : {};
      } catch {
        pollData = { error: "Server returned a non-JSON poll response.", details: pollRawText.slice(0, 500) };
      }

      if (!pollResponse.ok) {
        throw new Error(parseApiError(pollData));
      }

      const status = pollData.status || "unknown";
      if (status === "queued") {
        setStatus("loading", "Queued...");
      } else if (status === "running") {
        setStatus("loading", "Generating...");
      } else if (status === "completed") {
        outputEl.textContent = pollData.output || "(Empty output)";
        setStatus("done", "Done");
        break;
      } else if (status === "failed") {
        throw new Error(parseApiError(pollData));
      }

      if (Date.now() - startedAt > maxWaitMs) {
        throw new Error("Timed out waiting for generation result. Please try again.");
      }
    }
  } catch (error) {
    outputEl.textContent = String(error.message || error);
    setStatus("error", "Error");
  } finally {
    submitBtn.disabled = false;
  }
});
