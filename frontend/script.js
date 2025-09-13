// Farmer predict form
const form = document.getElementById("predictForm");
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = {};
    new FormData(form).forEach((v,k) => data[k] = isNaN(v) ? v : Number(v));
    const model = data["model"] || "rf";
    // send as { features: {...}, model: 'rf' }
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ features: data, model })
    });
    const j = await res.json();
    document.getElementById("result").innerText = "Predicted yield: " + (j.predicted_yield || j.detail || "");
  });
}

// Chatbot
const send = document.getElementById("send");
if (send) {
  send.addEventListener("click", async () => {
    const msg = document.getElementById("msg").value;
    const lang = document.getElementById("lang").value;
    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message: msg, lang })
    });
    const j = await res.json();
    document.getElementById("chatResp").innerText = j.reply;
  });
}
