// public/chatbot/chatbot.js
(() => {
  const toggle = document.getElementById("chatbot-toggle");
  const modal = document.getElementById("chatbot-modal");
  const closeBtn = document.getElementById("chatbot-close");
  const body = document.getElementById("chatbot-body");
  const form = document.getElementById("chatbot-form");
  const input = document.getElementById("chatbot-input");
  const sendBtn = document.getElementById("chatbot-send");

  function sanitize(s) {
    if (!s) return "";
    return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#039;"}[m]));
  }

  function formatTime(sec) {
    sec = Math.floor(sec||0);
    const h = Math.floor(sec/3600);
    const m = Math.floor((sec%3600)/60);
    const s = sec%60;
    if (h>0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    return `${m}:${String(s).padStart(2,'0')}`;
  }

  function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "chat-msg user";
    el.innerText = text;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function addBotMessage(text, sources) {
    const el = document.createElement("div");
    el.className = "chat-msg bot";
    const main = document.createElement("div");
    main.innerHTML = sanitize(text).replace(/\n/g,"<br/>");
    el.appendChild(main);

    if (Array.isArray(sources) && sources.length) {
      const chips = document.createElement("div");
      chips.className = "chat-sources";
      sources.forEach(s => {
        const c = document.createElement("div");
        c.className = "chat-source";
        if (s.video_link) {
          c.innerHTML = `<strong>${sanitize(s.title)}</strong> <a href="${sanitize(s.video_link)}" target="_blank" style="margin-left:8px;color:#cfe9ff;text-decoration:none">▶</a> <span style="margin-left:6px;opacity:.9">${formatTime(s.start)}</span>`;
        } else {
          c.innerHTML = `<strong>${sanitize(s.title)}</strong> <span style="margin-left:6px;opacity:.9">${formatTime(s.start)}</span>`;
        }
        chips.appendChild(c);
      });
      el.appendChild(chips);
    }

    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function addTyping() {
    const el = document.createElement("div");
    el.className = "chat-msg bot typing";
    el.id = "chatbot-typing";
    el.innerHTML = `<div class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById("chatbot-typing");
    if (t) t.remove();
  }

  toggle.addEventListener("click", () => {
    modal.classList.toggle("visible");
    if (modal.classList.contains("visible")) {
      modal.classList.remove("hidden");
      modal.setAttribute("aria-hidden", "false");
      setTimeout(() => input.focus(), 120);
      // if chat is empty, show intro
      if (body.children.length === 0) {
        addBotMessage("Hi — ask me anything about the course. I can point to the exact video & timestamp.", []);
      }
    } else {
      modal.classList.add("hidden");
      modal.setAttribute("aria-hidden", "true");
    }
  });

  closeBtn.addEventListener("click", () => {
    modal.classList.remove("visible");
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const txt = input.value.trim();
    if (!txt) return;
    addUserMessage(txt);
    input.value = "";
    sendBtn.disabled = true;
    addTyping();

    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: txt })
      });
      const json = await resp.json();
      removeTyping();
      sendBtn.disabled = false;
      if (json.error) {
        addBotMessage("Error: " + json.error, []);
        return;
      }
      addBotMessage(json.answer || "(no answer)", json.sources || []);
    } catch (err) {
      removeTyping();
      sendBtn.disabled = false;
      addBotMessage("Network error: " + err.message, []);
    }
  });

  // close modal when clicking outside panel
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.remove("visible");
      modal.classList.add("hidden");
      modal.setAttribute("aria-hidden", "true");
    }
  });
})();
