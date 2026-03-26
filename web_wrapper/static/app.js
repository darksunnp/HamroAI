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
  outputEl.textContent = "Calling Hugging Face Space...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        max_new_tokens: Number(tokenEl.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      let message = data.error || "Unknown server error";
      if (data.hint) {
        message += `\n\nHint: ${data.hint}`;
      }
      if (data.details) {
        message += `\n\nDetails: ${data.details}`;
      }
      throw new Error(message);
    }

    outputEl.textContent = data.output || "(Empty output)";
    setStatus("done", "Done");
  } catch (error) {
    outputEl.textContent = String(error.message || error);
    setStatus("error", "Error");
  } finally {
    submitBtn.disabled = false;
  }
});
