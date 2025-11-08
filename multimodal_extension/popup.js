document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const output = document.getElementById("output");
  output.textContent = "🔍 Analyzing current page... please wait.";

  // Get the active tab's URL
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab.url;

  try {
    // Send POST request to Flask backend
    const response = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    const result = await response.json();

    if (result.error) {
      output.textContent = "❌ Error: " + result.error;
    } else {
      output.innerHTML = `
        <strong>URL:</strong> ${result.url || url}<br>
        <strong>Emotional Impact:</strong> ${result.sentiment_raw?.emotional_impact || "N/A"}<br>
        <strong>Sentiment Type:</strong> ${result.sentiment_raw?.emotion || "N/A"}<br>
        <strong>Emotion Intensity:</strong> ${result.sentiment_raw?.intensity || "N/A"}<br>
        <strong>Keywords:</strong> ${result.keywords?.join(", ") || "N/A"}<br>
        <br>
        <strong>Snippet:</strong><br>
        <div style="color:#444;">${result.text_snippet || "No text extracted."}</div>
      `;
    }
  } catch (err) {
    output.textContent = "⚠️ Failed to connect to backend: " + err;
  }
});
