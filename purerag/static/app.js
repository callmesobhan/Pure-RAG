/**
 * Pure RAG — Clean, Uncluttered Frontend Application
 * Features:
 * - Smoothly Collapsible Left Sidebar on Desktop & Mobile
 * - Readable Answers with Bracketed Number Citations `[1]`, `[2]` + Hover Tooltips + Expandable References List
 * - Integrated Promptability Toolbar (Length pills + Style dropdown + Sources selector)
 * - Voice-to-Speech (STT) & Read-Aloud (TTS)
 * - Local Model Auto-Discovery & Hardware Setup Modal
 */

const state = {
  currentSessionId: null,
  chatHistory: [],
  selectedLength: "balanced",
  selectedStyle: "standard",
  topK: 6,
  customInstruction: "",
  selectedFilenames: new Set(),
  isStreaming: false,
  hardwareData: null,
  selectedTierId: "balanced",
  recognition: null,
  isRecording: false,
};

// ==================== DOM ELEMENTS ====================
const sidebar = document.getElementById("sidebar");
const closeSidebarBtn = document.getElementById("closeSidebarBtn");
const openSidebarBtn = document.getElementById("openSidebarBtn");
const docListEl = document.getElementById("docList");
const docCountBadge = document.getElementById("docCountBadge");
const activeModelDisplay = document.getElementById("activeModelDisplay");
const messagesContainer = document.getElementById("messagesContainer");
const messagesList = document.getElementById("messagesList");
const welcomeHero = document.getElementById("welcomeHero");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const voiceInputBtn = document.getElementById("voiceInputBtn");
const voiceStatusLabel = document.getElementById("voiceStatusLabel");
const autoSpeakToggle = document.getElementById("autoSpeakToggle");
const activeFilterNotice = document.getElementById("activeFilterNotice");
const styleSelect = document.getElementById("styleSelect");
const topKSelect = document.getElementById("topKSelect");

// ==================== SIDEBAR COLLAPSE / EXPAND ====================
function toggleSidebar(forceOpen) {
  if (typeof forceOpen === "boolean") {
    sidebar.classList.toggle("collapsed", !forceOpen);
  } else {
    sidebar.classList.toggle("collapsed");
  }
}

closeSidebarBtn.addEventListener("click", () => toggleSidebar(false));
openSidebarBtn.addEventListener("click", () => toggleSidebar());

// ==================== MARKDOWN & BRACKETED [1] CITATION RENDERER ====================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdownWithBracketCitations(rawText, citations = []) {
  if (!rawText) return "";
  let html = escapeHtml(rawText);

  // Build lookup map of rank -> citation metadata for tooltips
  const citeMap = {};
  for (const c of citations) {
    citeMap[c.rank] = c;
  }

  // Convert legacy verbose citations like `[filename.md, p. 1]` into bracketed numbers `[1]` if any LLM outputs them
  html = html.replace(/`?\[([^\]\n]+?),\s*p\.\s*(\d+)[^\]\n]*\]`?/gi, (_, fname, pageStr) => {
    const pageNum = parseInt(pageStr, 10);
    const matched = citations.find(
      (c) => c.filename.toLowerCase().includes(fname.trim().toLowerCase()) && c.page_number === pageNum
    ) || citations[0];
    const num = matched ? matched.rank : 1;
    return `[${num}]`;
  });

  // Code blocks ```...```
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  // Normalize grouped bracketed numbers like `[1, 2, 3]` or `[1, 2]` into `[1][2][3]` first
  html = html.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})+)\]/g, (_, groupStr) => {
    return groupStr
      .split(",")
      .map((s) => `[${s.trim()}]`)
      .join("");
  });

  // Replace bracketed numbers `[1]`, `[2]` with clean interactive badge pills
  html = html.replace(/\[(\d{1,2})\]/g, (full, numStr) => {
    const num = parseInt(numStr, 10);
    const meta = citeMap[num];
    const tooltip = meta
      ? `${meta.filename} (Page ${meta.page_number} — ${meta.section_path})`
      : `Source [${num}]`;
    return `<span class="cite-ref" data-ref="${num}" title="${escapeHtml(tooltip)}">[${num}]</span>`;
  });

  // Inline code
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

  // Process line-by-line into clean paragraphs, headings, lists, and blockquotes
  const lines = html.split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;

  const closeLists = () => {
    if (inUl) { out.push("</ul>"); inUl = false; }
    if (inOl) { out.push("</ol>"); inOl = false; }
  };

  for (let line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeLists();
      continue;
    }
    if (trimmed.startsWith("### ") || trimmed.startsWith("## ") || trimmed.startsWith("# ")) {
      closeLists();
      const cleanHeading = trimmed.replace(/^#{1,3}\s+/, "");
      out.push(`<h3>${cleanHeading}</h3>`);
    } else if (trimmed.startsWith("&gt; ")) {
      closeLists();
      out.push(`<blockquote>${trimmed.slice(5)}</blockquote>`);
    } else if (/^[-*•]\s+/.test(trimmed)) {
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (!inUl) { out.push("<ul>"); inUl = true; }
      out.push(`<li>${trimmed.replace(/^[-*•]\s+/, "")}</li>`);
    } else if (/^\d+\.\s+/.test(trimmed)) {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (!inOl) { out.push("<ol>"); inOl = true; }
      out.push(`<li>${trimmed.replace(/^\d+\.\s+/, "")}</li>`);
    } else {
      closeLists();
      out.push(`<p>${trimmed}</p>`);
    }
  }
  closeLists();
  return out.join("\n");
}

// ==================== ENGINE & DOCUMENT STATUS ====================
async function refreshStatus() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    renderDocumentList(data);
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }
}

function renderDocumentList(status) {
  const docs = status.documents || [];
  docCountBadge.textContent = docs.length;
  activeModelDisplay.textContent = `${status.llm_model || "Auto"} • ${status.total_chunks || 0} chunks`;

  if (docs.length === 0) {
    docListEl.innerHTML = `<div class="empty-docs">No documents in <code>pdfs/</code> yet.</div>`;
    return;
  }

  docListEl.innerHTML = "";
  for (const doc of docs) {
    const isSelected = state.selectedFilenames.has(doc.filename);
    const item = document.createElement("div");
    item.className = `doc-item ${isSelected ? "selected-filter" : ""}`;
    item.innerHTML = `
      <div class="doc-item-left" title="Click to filter answers to ${escapeHtml(doc.filename)}">
        <span>📄</span>
        <div class="doc-info">
          <span class="doc-name">${escapeHtml(doc.filename)}</span>
          <span class="doc-meta">${doc.total_pages}p • ${doc.chunk_count} chunks</span>
        </div>
      </div>
      <button class="doc-delete-btn" title="Remove file" aria-label="Delete ${escapeHtml(doc.filename)}">✕</button>
    `;

    item.querySelector(".doc-item-left").addEventListener("click", () => {
      if (state.selectedFilenames.has(doc.filename)) {
        state.selectedFilenames.delete(doc.filename);
      } else {
        state.selectedFilenames.add(doc.filename);
      }
      updateFilterNotice();
      renderDocumentList(status);
    });

    item.querySelector(".doc-delete-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove "${doc.filename}" from pdfs/?`)) return;
      await fetch(`/api/documents/${encodeURIComponent(doc.filename)}`, { method: "DELETE" });
      state.selectedFilenames.delete(doc.filename);
      updateFilterNotice();
      await refreshStatus();
    });

    docListEl.appendChild(item);
  }
}

function updateFilterNotice() {
  if (state.selectedFilenames.size === 0) {
    activeFilterNotice.classList.add("hidden");
    activeFilterNotice.textContent = "";
  } else {
    activeFilterNotice.classList.remove("hidden");
    activeFilterNotice.textContent = `Filtered: ${Array.from(state.selectedFilenames).join(", ")}`;
  }
}

// ==================== FILE UPLOAD & FOLDER SYNC ====================
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const syncDocsBtn = document.getElementById("syncDocsBtn");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files?.length) {
    await uploadFiles(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", async () => {
  if (fileInput.files?.length) {
    await uploadFiles(fileInput.files);
    fileInput.value = "";
  }
});

async function uploadFiles(fileList) {
  const formData = new FormData();
  for (const f of fileList) {
    formData.append("files", f);
  }
  dropzone.querySelector("strong").textContent = "Indexing...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.engine) renderDocumentList(data.engine);
  } finally {
    dropzone.querySelector("strong").textContent = "Add PDFs or Docs";
  }
}

syncDocsBtn.addEventListener("click", async () => {
  syncDocsBtn.textContent = "Syncing...";
  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    const data = await res.json();
    if (data.engine) renderDocumentList(data.engine);
  } finally {
    syncDocsBtn.textContent = "Sync";
  }
});

// ==================== PROMPTABILITY CONTROLS ====================
document.querySelectorAll("#lengthSelector .pill-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#lengthSelector .pill-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.selectedLength = btn.dataset.value;
  });
});

styleSelect.addEventListener("change", (e) => {
  state.selectedStyle = e.target.value;
});

topKSelect.addEventListener("change", (e) => {
  state.topK = parseInt(e.target.value, 10) || 6;
});

function setPromptabilityUI(lengthVal, styleVal) {
  if (lengthVal) {
    state.selectedLength = lengthVal;
    document.querySelectorAll("#lengthSelector .pill-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.value === lengthVal);
    });
  }
  if (styleVal) {
    state.selectedStyle = styleVal;
    styleSelect.value = styleVal;
  }
}

document.querySelectorAll(".starter-card").forEach((card) => {
  card.addEventListener("click", () => {
    setPromptabilityUI(card.dataset.length, card.dataset.style);
    promptInput.value = card.dataset.prompt;
    sendMessage();
  });
});

// ==================== VOICE-TO-SPEECH (STT) & TEXT-TO-SPEECH (TTS) ====================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = navigator.language || "en-US";

  let basePrompt = "";
  recognition.onstart = () => {
    state.isRecording = true;
    basePrompt = promptInput.value ? promptInput.value + " " : "";
    voiceInputBtn.classList.add("recording");
    voiceStatusLabel.textContent = "Listening...";
  };
  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    promptInput.value = basePrompt + transcript;
    autoResizeTextarea();
  };
  recognition.onerror = () => stopVoiceRecording();
  recognition.onend = () => stopVoiceRecording();
  state.recognition = recognition;
}

function stopVoiceRecording() {
  state.isRecording = false;
  voiceInputBtn.classList.remove("recording");
  voiceStatusLabel.textContent = "Voice";
}

voiceInputBtn.addEventListener("click", () => {
  if (!state.recognition) initSpeechRecognition();
  if (!state.recognition) {
    alert("Speech recognition is supported in Chrome, Edge, and Safari.");
    return;
  }
  if (state.isRecording) state.recognition.stop();
  else state.recognition.start();
});

function speakTextAloud(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const clean = text
    .replace(/\[\d+\]/g, "")
    .replace(/[#*_>`~-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!clean) return;
  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.rate = 1.03;
  window.speechSynthesis.speak(utterance);
}

// ==================== CHAT STREAMING & CLEAN NUMBERED REFERENCES ====================
function autoResizeTextarea() {
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + "px";
}

promptInput.addEventListener("input", autoResizeTextarea);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", () => sendMessage());

document.getElementById("newChatBtn").addEventListener("click", () => {
  state.currentSessionId = null;
  state.chatHistory = [];
  messagesList.innerHTML = "";
  welcomeHero.style.display = "block";
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  refreshHistorySidebar();
});

async function sendMessage() {
  const question = promptInput.value.trim();
  if (!question || state.isStreaming) return;

  welcomeHero.style.display = "none";
  promptInput.value = "";
  autoResizeTextarea();
  state.isStreaming = true;

  // 1. User Bubble
  const userRow = document.createElement("div");
  userRow.className = "message-row user";
  userRow.innerHTML = `<div class="message-bubble">${escapeHtml(question)}</div>`;
  messagesList.appendChild(userRow);

  // 2. Clean Assistant Answer Block (No Cluttered Top Header!)
  const assistantRow = document.createElement("div");
  assistantRow.className = "message-row assistant";
  assistantRow.innerHTML = `
    <div class="message-bubble">
      <div class="msg-content"><p><em>Thinking...</em></p></div>
      <div class="msg-footer-bar">
        <div class="msg-actions">
          <button class="msg-action-btn toggle-refs-btn">📚 References (0)</button>
          <button class="msg-action-btn speak-btn" title="Read response aloud">🔊 Speak</button>
          <button class="msg-action-btn copy-btn" title="Copy answer">📋 Copy</button>
        </div>
        <span class="msg-meta-info"></span>
      </div>
      <div class="references-box hidden"></div>
    </div>
  `;
  messagesList.appendChild(assistantRow);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  const contentEl = assistantRow.querySelector(".msg-content");
  const metaInfoEl = assistantRow.querySelector(".msg-meta-info");
  const toggleRefsBtn = assistantRow.querySelector(".toggle-refs-btn");
  const referencesBoxEl = assistantRow.querySelector(".references-box");
  const speakBtn = assistantRow.querySelector(".speak-btn");
  const copyBtn = assistantRow.querySelector(".copy-btn");

  let accumulatedAnswer = "";
  let citations = [];

  toggleRefsBtn.addEventListener("click", () => {
    referencesBoxEl.classList.toggle("hidden");
  });

  // Clicking any inline `[1]` badge opens the references list and highlights that reference
  contentEl.addEventListener("click", (e) => {
    const badge = e.target.closest(".cite-ref");
    if (!badge) return;
    referencesBoxEl.classList.remove("hidden");
    const refItem = referencesBoxEl.querySelector(`[data-ref-item="${badge.dataset.ref}"]`);
    if (refItem) {
      refItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });

  try {
    const payload = {
      question,
      session_id: state.currentSessionId,
      length: state.selectedLength,
      style: state.selectedStyle,
      top_k: state.topK,
      custom_instruction: state.customInstruction || null,
      filter_filenames: state.selectedFilenames.size > 0 ? Array.from(state.selectedFilenames) : null,
      chat_history: state.chatHistory.slice(-6),
    };

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        const evt = JSON.parse(jsonStr);
        if (evt.type === "metadata") {
          citations = evt.citations || [];
          toggleRefsBtn.textContent = `📚 References (${citations.length})`;
          renderNumberedReferences(referencesBoxEl, citations);
        } else if (evt.type === "token") {
          accumulatedAnswer += evt.content || "";
          contentEl.innerHTML = renderMarkdownWithBracketCitations(accumulatedAnswer, citations);
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        } else if (evt.type === "done") {
          if (evt.session_id) {
            state.currentSessionId = evt.session_id;
          }
          accumulatedAnswer = evt.answer || accumulatedAnswer;
          contentEl.innerHTML = renderMarkdownWithBracketCitations(accumulatedAnswer, citations);
          const pm = evt.pipeline_metrics || {};
          const retMs = pm.retrieval_ms ?? 0;
          const genMs = evt.generation_ms ?? pm.generation_ms ?? 0;
          metaInfoEl.textContent = `${state.selectedLength} • ${state.selectedStyle} • ${evt.generator || ""} (${retMs}ms + ${genMs}ms)`;
          refreshHistorySidebar();
        }
      }
    }

    state.chatHistory.push({ role: "user", content: question });
    state.chatHistory.push({ role: "assistant", content: accumulatedAnswer });

    speakBtn.addEventListener("click", () => speakTextAloud(accumulatedAnswer));
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(accumulatedAnswer);
      copyBtn.textContent = "✓ Copied";
      setTimeout(() => (copyBtn.textContent = "📋 Copy"), 1500);
    });

    if (autoSpeakToggle.checked) {
      speakTextAloud(accumulatedAnswer);
    }
  } catch (err) {
    contentEl.innerHTML = `<p style="color:#f43f5e;">Error generating answer: ${escapeHtml(err.message)}</p>`;
  } finally {
    state.isStreaming = false;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function renderNumberedReferences(container, citations) {
  if (!citations || citations.length === 0) {
    container.innerHTML = "";
    return;
  }
  const itemsHtml = citations
    .map(
      (c) => `
      <div class="ref-item" data-ref-item="${c.rank}">
        <span class="ref-num">[${c.rank}]</span>
        <div class="ref-body">
          <span class="ref-title">${escapeHtml(c.filename)}</span>
          <span class="ref-meta">Page ${c.page_number} • ${escapeHtml(c.section_path)}</span>
          <span class="ref-snippet">${escapeHtml(c.snippet)}</span>
        </div>
      </div>
    `
    )
    .join("");

  container.innerHTML = `<div class="references-list">${itemsHtml}</div>`;
}

// ==================== HARDWARE & LOCAL MODELS MODAL ====================
const hardwareModal = document.getElementById("hardwareModal");
const openHardwareModalBtn = document.getElementById("openHardwareModalBtn");
const closeHardwareModalBtn = document.getElementById("closeHardwareModalBtn");
const tiersGrid = document.getElementById("tiersGrid");
const setupLlmProvider = document.getElementById("setupLlmProvider");
const setupLlmModel = document.getElementById("setupLlmModel");
const setupEmbeddingBackend = document.getElementById("setupEmbeddingBackend");
const setupUseReranker = document.getElementById("setupUseReranker");
const approveDownloadCheckbox = document.getElementById("approveDownloadCheckbox");

openHardwareModalBtn.addEventListener("click", () => loadAndOpenHardwareModal());
closeHardwareModalBtn.addEventListener("click", () => hardwareModal.classList.add("hidden"));

async function loadAndOpenHardwareModal() {
  hardwareModal.classList.remove("hidden");
  try {
    const res = await fetch("/api/hardware");
    const data = await res.json();
    state.hardwareData = data;
    const hw = data.hardware;
    const cfg = data.config;

    document.getElementById("specOs").textContent = `${hw.os_name} (${hw.architecture})`;
    document.getElementById("specCpu").textContent = `${hw.cpu_brand}`;
    document.getElementById("specRam").textContent = `${hw.ram_total_gb} GB`;
    document.getElementById("specGpu").textContent = hw.gpu_name;

    const rec = hw.recommendation_details;
    const disc = hw.discovered_local_models || {};
    document.getElementById("recTitle").textContent = `${rec.name} — Active LLM: ${rec.llm_model}`;
    document.getElementById("recReason").textContent = rec.why_recommended;

    const discBox = document.getElementById("discoveredModelsBox");
    if (discBox) {
      const llmList = (disc.ollama_llms || []).join(", ") || "None";
      const embList = [...(disc.ollama_embeddings || []), ...(disc.hf_embeddings || [])].join(", ") || "Built-in Hashed N-Gram";
      const rerankList = (disc.hf_rerankers || []).join(", ") || "Built-in Cross-Proximity";
      discBox.innerHTML = `
        <strong>Detected On-Device Models (Reused Automatically):</strong><br/>
        • LLMs: <code>${escapeHtml(llmList)}</code> &nbsp;|&nbsp;
        Embedders: <code>${escapeHtml(embList)}</code> &nbsp;|&nbsp;
        Rerankers: <code>${escapeHtml(rerankList)}</code>
      `;
    }

    state.selectedTierId = cfg.hardware_profiled ? cfg.selected_tier : hw.recommended_tier;
    renderTiersGrid(hw.all_tiers, state.selectedTierId, disc);

    const activeTier = hw.all_tiers[state.selectedTierId] || rec;
    setupLlmProvider.value = cfg.llm_provider || "ollama";
    setupLlmModel.value = disc.best_local_llm || cfg.llm_model || activeTier.llm_model;
    setupEmbeddingBackend.value = disc.best_local_embedder_backend || cfg.embedding_backend || activeTier.embedding_backend;
    setupUseReranker.checked = cfg.use_reranker ?? activeTier.reranker_enabled;
  } catch (err) {
    console.error("Error loading hardware profile:", err);
  }
}

function renderTiersGrid(allTiers, selectedId, disc = {}) {
  tiersGrid.innerHTML = "";
  for (const [tierId, tier] of Object.entries(allTiers)) {
    const card = document.createElement("div");
    card.className = `tier-card ${tierId === selectedId ? "selected" : ""}`;
    card.innerHTML = `
      <div class="tier-card-top">
        <strong>${escapeHtml(tier.name)}</strong>
        <span class="tier-badge">${escapeHtml(tier.badge)}</span>
      </div>
      <div class="tier-models">LLM: ${escapeHtml(tier.llm_model)} • Embed: ${escapeHtml(tier.embedding_model)}</div>
      <div class="tier-desc">${escapeHtml(tier.description)}</div>
    `;
    card.addEventListener("click", () => {
      state.selectedTierId = tierId;
      setupLlmModel.value = disc.best_local_llm || tier.llm_model;
      setupEmbeddingBackend.value = disc.best_local_embedder_backend || tier.embedding_backend;
      setupUseReranker.checked = tier.reranker_enabled;
      renderTiersGrid(allTiers, tierId, disc);
    });
    tiersGrid.appendChild(card);
  }
}

document.getElementById("saveHardwareSetupBtn").addEventListener("click", async () => {
  const btn = document.getElementById("saveHardwareSetupBtn");
  btn.textContent = "Saving...";
  try {
    await fetch("/api/setup/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tier_id: state.selectedTierId,
        approved: approveDownloadCheckbox.checked,
        llm_provider: setupLlmProvider.value,
        llm_model: setupLlmModel.value.trim(),
        embedding_backend: setupEmbeddingBackend.value,
        use_reranker: setupUseReranker.checked,
      }),
    });
    hardwareModal.classList.add("hidden");
    await refreshStatus();
  } finally {
    btn.textContent = "Save & Apply";
  }
});

document.getElementById("keepOfflineFallbackBtn").addEventListener("click", async () => {
  await fetch("/api/setup/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tier_id: "ultra_light",
      approved: false,
      llm_provider: "extractive_fallback",
      embedding_backend: "hashed_ngram",
      use_reranker: true,
    }),
  });
  hardwareModal.classList.add("hidden");
  await refreshStatus();
});

// ==================== DOMAIN PERSONA MODAL ====================
const personaModal = document.getElementById("personaModal");
const customPersonaInput = document.getElementById("customPersonaInput");

document.getElementById("openCustomPersonaBtn").addEventListener("click", () => {
  personaModal.classList.remove("hidden");
});
document.getElementById("closePersonaModalBtn").addEventListener("click", () => {
  personaModal.classList.add("hidden");
});

document.querySelectorAll(".preset-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    customPersonaInput.value = chip.dataset.persona || "";
  });
});

document.getElementById("savePersonaBtn").addEventListener("click", async () => {
  state.customInstruction = customPersonaInput.value.trim();
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ custom_system_prompt: state.customInstruction }),
  });
  personaModal.classList.add("hidden");
});

// ==================== Q&A HISTORY & METADATA MANAGER ====================
const sessionListEl = document.getElementById("sessionList");
const historyCountBadge = document.getElementById("historyCountBadge");
const historyModal = document.getElementById("historyModal");
const openHistoryModalBtn = document.getElementById("openHistoryModalBtn");
const topbarHistoryBtn = document.getElementById("topbarHistoryBtn");
const closeHistoryModalBtn = document.getElementById("closeHistoryModalBtn");
const historySearchInput = document.getElementById("historySearchInput");
const historyStatsBar = document.getElementById("historyStatsBar");
const historyTurnsList = document.getElementById("historyTurnsList");
const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");

async function refreshHistorySidebar() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const sessions = data.sessions || [];
    if (historyCountBadge) historyCountBadge.textContent = data.total_turns ?? sessions.length;
    if (!sessionListEl) return;

    if (sessions.length === 0) {
      sessionListEl.innerHTML = `<div class="empty-docs">No past chats yet.</div>`;
      return;
    }

    sessionListEl.innerHTML = "";
    for (const s of sessions) {
      const isCurrent = s.session_id === state.currentSessionId;
      const item = document.createElement("div");
      item.className = `session-item ${isCurrent ? "active-session" : ""}`;
      const timeStr = (s.updated_at_iso || "").slice(0, 16).replace("T", " ");
      item.innerHTML = `
        <div class="session-info" title="${escapeHtml(s.title)}">
          <span class="session-title">💬 ${escapeHtml(s.title)}</span>
          <span class="session-meta">${s.turn_count} Q&A • ${escapeHtml(timeStr)}</span>
        </div>
        <button class="doc-delete-btn" title="Delete session" aria-label="Delete session">✕</button>
      `;

      item.querySelector(".session-info").addEventListener("click", () => {
        loadSavedSessionIntoChat(s.session_id);
      });

      item.querySelector(".doc-delete-btn").addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/history/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
        if (state.currentSessionId === s.session_id) {
          state.currentSessionId = null;
          state.chatHistory = [];
          messagesList.innerHTML = "";
          welcomeHero.style.display = "block";
        }
        await refreshHistorySidebar();
      });

      sessionListEl.appendChild(item);
    }
  } catch (err) {
    console.error("Failed to load history sidebar:", err);
  }
}

async function loadSavedSessionIntoChat(sessionId) {
  try {
    const res = await fetch(`/api/history/${encodeURIComponent(sessionId)}`);
    if (!res.ok) return;
    const sess = await res.json();
    state.currentSessionId = sess.session_id;
    state.chatHistory = [];
    messagesList.innerHTML = "";
    welcomeHero.style.display = "none";

    for (const t of sess.turns || []) {
      // User bubble
      const userRow = document.createElement("div");
      userRow.className = "message-row user";
      userRow.innerHTML = `<div class="message-bubble">${escapeHtml(t.question)}</div>`;
      messagesList.appendChild(userRow);

      // Assistant bubble
      const cites = t.citations || [];
      const pr = t.promptability || {};
      const pm = t.pipeline_metrics || {};
      const assistantRow = document.createElement("div");
      assistantRow.className = "message-row assistant";
      assistantRow.innerHTML = `
        <div class="message-bubble">
          <div class="msg-content">${renderMarkdownWithBracketCitations(t.answer, cites)}</div>
          <div class="msg-footer-bar">
            <div class="msg-actions">
              <button class="msg-action-btn toggle-refs-btn">📚 References (${cites.length})</button>
              <button class="msg-action-btn speak-btn">🔊 Speak</button>
              <button class="msg-action-btn copy-btn">📋 Copy</button>
            </div>
            <span class="msg-meta-info">${escapeHtml(pr.length || "balanced")} • ${escapeHtml(pr.style || "standard")} • ${escapeHtml(pm.generator || "")} (${pm.retrieval_ms ?? 0}ms + ${pm.generation_ms ?? 0}ms)</span>
          </div>
          <div class="references-box hidden"></div>
        </div>
      `;
      const refsBox = assistantRow.querySelector(".references-box");
      renderNumberedReferences(refsBox, cites);

      assistantRow.querySelector(".toggle-refs-btn").addEventListener("click", () => {
        refsBox.classList.toggle("hidden");
      });
      assistantRow.querySelector(".msg-content").addEventListener("click", (e) => {
        const badge = e.target.closest(".cite-ref");
        if (!badge) return;
        refsBox.classList.remove("hidden");
        const refItem = refsBox.querySelector(`[data-ref-item="${badge.dataset.ref}"]`);
        if (refItem) refItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
      assistantRow.querySelector(".speak-btn").addEventListener("click", () => speakTextAloud(t.answer));
      const copyBtn = assistantRow.querySelector(".copy-btn");
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(t.answer);
        copyBtn.textContent = "✓ Copied";
        setTimeout(() => (copyBtn.textContent = "📋 Copy"), 1500);
      });

      messagesList.appendChild(assistantRow);
      state.chatHistory.push({ role: "user", content: t.question });
      state.chatHistory.push({ role: "assistant", content: t.answer });
    }
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    await refreshHistorySidebar();
  } catch (err) {
    console.error("Failed to restore session:", err);
  }
}

async function loadAndOpenHistoryModal(searchQuery = "") {
  historyModal.classList.remove("hidden");
  try {
    const url = searchQuery
      ? `/api/history?q=${encodeURIComponent(searchQuery)}`
      : "/api/history";
    const res = await fetch(url);
    const data = await res.json();
    const turns = data.turns || [];
    historyStatsBar.textContent = `Showing ${turns.length} Q&A turn(s) across ${data.total_sessions || 0} conversation session(s) • Stored in data/chat_history.json`;

    if (turns.length === 0) {
      historyTurnsList.innerHTML = `<div class="empty-docs">No matching Q&A records found.</div>`;
      return;
    }

    historyTurnsList.innerHTML = "";
    for (const t of turns) {
      const pr = t.promptability || {};
      const pm = t.pipeline_metrics || {};
      const cites = t.citations || [];
      const timeStr = (t.created_at_iso || "").slice(0, 19).replace("T", " ");
      const citeSummary = cites
        .map((c) => `[${c.rank}] ${c.filename} (p.${c.page_number}, score:${c.final_score})`)
        .join("  •  ");

      const card = document.createElement("div");
      card.className = "history-turn-card";
      card.innerHTML = `
        <div class="history-turn-top">
          <div class="history-turn-badges">
            <span class="meta-pill">🕒 ${escapeHtml(timeStr)}</span>
            <span class="meta-pill">Length: ${escapeHtml(pr.length || "balanced")}</span>
            <span class="meta-pill">Style: ${escapeHtml(pr.style || "standard")}</span>
            <span class="meta-pill">Top-K: ${pr.top_k ?? 6}</span>
            <span class="meta-pill">LLM: ${escapeHtml(pm.generator || "N/A")}</span>
            <span class="meta-pill">Embed: ${escapeHtml(pm.embedder || "N/A")}</span>
            <span class="meta-pill">Rerank: ${escapeHtml(pm.reranker || "N/A")}</span>
            <span class="meta-pill">⏱️ ${pm.retrieval_ms ?? 0}ms ret / ${pm.generation_ms ?? 0}ms gen</span>
          </div>
          <div class="msg-actions">
            <button class="msg-action-btn open-in-chat-btn">↗ Open in Chat</button>
            <button class="msg-action-btn delete-turn-btn">🗑️ Delete</button>
          </div>
        </div>
        <div class="history-turn-q">Q: ${escapeHtml(t.question)}</div>
        <div class="history-turn-a">${renderMarkdownWithBracketCitations(t.answer, cites)}</div>
        <div class="history-turn-cites">📚 <strong>Citations (${cites.length}):</strong> ${escapeHtml(citeSummary || "None")}</div>
      `;

      card.querySelector(".open-in-chat-btn").addEventListener("click", async () => {
        historyModal.classList.add("hidden");
        await loadSavedSessionIntoChat(t.session_id);
      });

      card.querySelector(".delete-turn-btn").addEventListener("click", async () => {
        await fetch(
          `/api/history/${encodeURIComponent(t.session_id)}/turns/${encodeURIComponent(t.turn_id)}`,
          { method: "DELETE" }
        );
        await loadAndOpenHistoryModal(historySearchInput.value.trim());
        await refreshHistorySidebar();
      });

      historyTurnsList.appendChild(card);
    }
  } catch (err) {
    console.error("Failed to load history modal:", err);
  }
}

if (openHistoryModalBtn) {
  openHistoryModalBtn.addEventListener("click", () => loadAndOpenHistoryModal(historySearchInput.value.trim()));
}
if (topbarHistoryBtn) {
  topbarHistoryBtn.addEventListener("click", () => loadAndOpenHistoryModal(historySearchInput.value.trim()));
}
if (closeHistoryModalBtn) {
  closeHistoryModalBtn.addEventListener("click", () => historyModal.classList.add("hidden"));
}
if (historySearchInput) {
  historySearchInput.addEventListener("input", () => {
    loadAndOpenHistoryModal(historySearchInput.value.trim());
  });
}
if (clearAllHistoryBtn) {
  clearAllHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Delete all saved Q&A history and metadata?")) return;
    await fetch("/api/history", { method: "DELETE" });
    state.currentSessionId = null;
    await loadAndOpenHistoryModal("");
    await refreshHistorySidebar();
  });
}

// ==================== THEME TOGGLE ====================
document.getElementById("themeToggleBtn").addEventListener("click", () => {
  const html = document.documentElement;
  const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
});

// ==================== INITIALIZATION ====================
(async function initApp() {
  initSpeechRecognition();
  await refreshStatus();
  await refreshHistorySidebar();
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    state.customInstruction = cfg.custom_system_prompt || "";
    customPersonaInput.value = state.customInstruction;
  } catch (e) {
    console.error(e);
  }
})();
