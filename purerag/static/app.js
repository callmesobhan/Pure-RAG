/**
 * Pure RAG — Frontend Application Logic
 * Features:
 * - Streaming SSE Grounded Chat with Inline Page Citations & Source Cards
 * - Promptability Controls (Length: Shorter/Balanced/Longer, Style: Standard/Precise/Bulleted/Executive/Steps/ELI5/Quotes)
 * - Voice-to-Speech (Web Speech API STT Dictation) & Text-to-Speech (TTS Read-Aloud)
 * - Drag-and-Drop PDF Ingestion + Live Folder Sync
 * - Hardware Capability Analyzer & Permissioned Setup Wizard
 */

const state = {
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
const docListEl = document.getElementById("docList");
const docCountBadge = document.getElementById("docCountBadge");
const pdfsPathDisplay = document.getElementById("pdfsPathDisplay");
const activeModelDisplay = document.getElementById("activeModelDisplay");
const activeRetrieverDisplay = document.getElementById("activeRetrieverDisplay");
const chunkStatsDisplay = document.getElementById("chunkStatsDisplay");
const messagesContainer = document.getElementById("messagesContainer");
const messagesList = document.getElementById("messagesList");
const welcomeHero = document.getElementById("welcomeHero");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const voiceInputBtn = document.getElementById("voiceInputBtn");
const voiceStatusLabel = document.getElementById("voiceStatusLabel");
const autoSpeakToggle = document.getElementById("autoSpeakToggle");
const activeFilterNotice = document.getElementById("activeFilterNotice");

// ==================== MARKDOWN & CITATION RENDERER ====================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdownWithCitations(rawText) {
  if (!rawText) return "";
  let html = escapeHtml(rawText);

  // Code blocks ```...```
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  // Highlight inline citations like [filename.pdf, p. 4] or `[sample.md, p. 1]`
  html = html.replace(
    /`?\[([^\]\n]+?,\s*p\.\s*\d+[^\]\n]*)\]`?/gi,
    '<span class="inline-citation">📄 $1</span>'
  );

  // Inline code
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

  // Process line by line for headings, blockquotes, lists, paragraphs
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
    if (trimmed.startsWith("### ")) {
      closeLists();
      out.push(`<h3>${trimmed.slice(4)}</h3>`);
    } else if (trimmed.startsWith("## ")) {
      closeLists();
      out.push(`<h3>${trimmed.slice(3)}</h3>`);
    } else if (trimmed.startsWith("# ")) {
      closeLists();
      out.push(`<h3>${trimmed.slice(2)}</h3>`);
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

function formatBytes(bytes) {
  if (!bytes || bytes < 1024) return `${bytes || 0} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderDocumentList(status) {
  const docs = status.documents || [];
  docCountBadge.textContent = docs.length;
  pdfsPathDisplay.textContent = `📂 ${status.pdfs_dir || "./pdfs"}`;
  activeModelDisplay.textContent = `LLM: ${status.llm_model || "Auto"}`;
  activeRetrieverDisplay.textContent = `${status.active_embedder} + ${status.active_reranker}`;
  chunkStatsDisplay.textContent = `${status.total_chunks || 0} contextual chunks across ${docs.length} docs`;

  if (docs.length === 0) {
    docListEl.innerHTML = `
      <div class="empty-docs">
        No PDFs or docs indexed yet.<br/>
        Drop files above or place them in <code>pdfs/</code>!
      </div>
    `;
    return;
  }

  docListEl.innerHTML = "";
  for (const doc of docs) {
    const isSelected = state.selectedFilenames.has(doc.filename);
    const item = document.createElement("div");
    item.className = `doc-item ${isSelected ? "selected-filter" : ""}`;
    item.innerHTML = `
      <div class="doc-item-left" title="Click to filter queries to ${escapeHtml(doc.filename)}">
        <span>📄</span>
        <div class="doc-info">
          <span class="doc-name">${escapeHtml(doc.filename)}</span>
          <span class="doc-meta">${doc.total_pages}p • ${doc.chunk_count} chunks • ${formatBytes(doc.file_size_bytes)}</span>
        </div>
      </div>
      <button class="doc-delete-btn" title="Remove file from pdfs/ folder" aria-label="Delete ${escapeHtml(doc.filename)}">✕</button>
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
      if (!confirm(`Remove "${doc.filename}" from the pdfs/ folder?`)) return;
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
    activeFilterNotice.textContent = "";
  } else {
    activeFilterNotice.textContent = `Filtering (${state.selectedFilenames.size}): ${Array.from(state.selectedFilenames).join(", ")}`;
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
  dropzone.querySelector("strong").textContent = "Indexing Documents...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.engine) renderDocumentList(data.engine);
  } catch (err) {
    console.error("Upload failed:", err);
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
    syncDocsBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
        <polyline points="23 4 23 10 17 10"></polyline>
        <polyline points="1 20 1 14 7 14"></polyline>
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
      </svg>
      Sync Folder
    `;
  }
});

// ==================== PROMPTABILITY CONTROLS ====================
function bindSegmentedGroup(containerId, stateKey) {
  const container = document.getElementById(containerId);
  const buttons = container.querySelectorAll(".pill-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state[stateKey] = btn.dataset.value;
    });
  });
}

bindSegmentedGroup("lengthSelector", "selectedLength");
bindSegmentedGroup("styleSelector", "selectedStyle");

document.getElementById("topKSelect").addEventListener("change", (e) => {
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
    document.querySelectorAll("#styleSelector .pill-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.value === styleVal);
    });
  }
}

// Starter cards on Welcome Hero
document.querySelectorAll(".starter-card").forEach((card) => {
  card.addEventListener("click", () => {
    const promptText = card.dataset.prompt;
    const styleVal = card.dataset.style;
    const lengthVal = card.dataset.length;
    setPromptabilityUI(lengthVal, styleVal);
    promptInput.value = promptText;
    sendMessage();
  });
});

// ==================== VOICE-TO-SPEECH (STT) & TEXT-TO-SPEECH (TTS) ====================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceInputBtn.title = "Speech recognition is supported in Chrome, Edge, and Safari.";
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = navigator.language || "en-US";

  let basePromptBeforeVoice = "";

  recognition.onstart = () => {
    state.isRecording = true;
    basePromptBeforeVoice = promptInput.value ? promptInput.value + " " : "";
    voiceInputBtn.classList.add("recording");
    voiceStatusLabel.textContent = "Listening...";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    promptInput.value = basePromptBeforeVoice + transcript;
    autoResizeTextarea();
  };

  recognition.onerror = () => {
    stopVoiceRecording();
  };

  recognition.onend = () => {
    stopVoiceRecording();
  };

  state.recognition = recognition;
}

function stopVoiceRecording() {
  state.isRecording = false;
  voiceInputBtn.classList.remove("recording");
  voiceStatusLabel.textContent = "Voice";
}

voiceInputBtn.addEventListener("click", () => {
  if (!state.recognition) {
    initSpeechRecognition();
  }
  if (!state.recognition) {
    alert("Browser Voice-to-Text (Web Speech API) is not available in this browser. Try Chrome, Edge, or Safari.");
    return;
  }
  if (state.isRecording) {
    state.recognition.stop();
  } else {
    state.recognition.start();
  }
});

function speakTextAloud(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const clean = text
    .replace(/`?\[[^\]\n]+?,\s*p\.\s*\d+[^\]\n]*\]`?/gi, "")
    .replace(/[#*_>`~-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!clean) return;
  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.rate = 1.03;
  window.speechSynthesis.speak(utterance);
}

// ==================== CHAT STREAMING & UI RENDERING ====================
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
  state.chatHistory = [];
  messagesList.innerHTML = "";
  welcomeHero.style.display = "block";
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
});

async function sendMessage() {
  const question = promptInput.value.trim();
  if (!question || state.isStreaming) return;

  welcomeHero.style.display = "none";
  promptInput.value = "";
  autoResizeTextarea();
  state.isStreaming = true;

  // 1. Append User Bubble
  const userRow = document.createElement("div");
  userRow.className = "message-row user";
  userRow.innerHTML = `<div class="message-bubble">${escapeHtml(question)}</div>`;
  messagesList.appendChild(userRow);

  // 2. Append Assistant Bubble Skeleton
  const assistantRow = document.createElement("div");
  assistantRow.className = "message-row assistant";
  assistantRow.innerHTML = `
    <div class="message-bubble">
      <div class="msg-header-bar">
        <div class="msg-badges">
          <span class="msg-badge">Length: ${escapeHtml(state.selectedLength)}</span>
          <span class="msg-badge">Style: ${escapeHtml(state.selectedStyle)}</span>
          <span class="msg-badge" id="streamMetricBadge">Retrieving Hybrid + Rerank...</span>
        </div>
        <div class="msg-actions">
          <button class="msg-action-btn speak-btn" title="Read response aloud (Text-to-Speech)">🔊 Speak</button>
          <button class="msg-action-btn copy-btn" title="Copy Markdown response">📋 Copy</button>
        </div>
      </div>
      <div class="msg-content"><p><em>Searching hybrid index &amp; reranking chunks...</em></p></div>
      <div class="citations-panel"></div>
    </div>
  `;
  messagesList.appendChild(assistantRow);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  const contentEl = assistantRow.querySelector(".msg-content");
  const metricBadgeEl = assistantRow.querySelector("#streamMetricBadge");
  const citationsPanelEl = assistantRow.querySelector(".citations-panel");
  const speakBtn = assistantRow.querySelector(".speak-btn");
  const copyBtn = assistantRow.querySelector(".copy-btn");

  let accumulatedAnswer = "";
  let citations = [];

  try {
    const payload = {
      question,
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
          const retMs = evt.pipeline_metrics?.retrieval_ms ?? 0;
          metricBadgeEl.textContent = `Hybrid RRF + Rerank: ${retMs}ms • ${citations.length} sources`;
          renderCitationsDrawer(citationsPanelEl, citations);
        } else if (evt.type === "token") {
          accumulatedAnswer += evt.content || "";
          contentEl.innerHTML = renderMarkdownWithCitations(accumulatedAnswer);
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        } else if (evt.type === "done") {
          accumulatedAnswer = evt.answer || accumulatedAnswer;
          contentEl.innerHTML = renderMarkdownWithCitations(accumulatedAnswer);
          metricBadgeEl.textContent += ` • ${evt.generator}`;
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

function renderCitationsDrawer(container, citations) {
  if (!citations || citations.length === 0) {
    container.innerHTML = "";
    return;
  }
  const cardsHtml = citations
    .map(
      (c) => `
      <div class="citation-card">
        <div class="citation-card-header">
          <span>#${c.rank} ${escapeHtml(c.filename)} (p. ${c.page_number})</span>
          <span>Score: ${c.final_score}</span>
        </div>
        <div class="citation-snippet" title="${escapeHtml(c.full_text)}">
          <strong>${escapeHtml(c.section_path)}:</strong> ${escapeHtml(c.snippet)}
        </div>
      </div>
    `
    )
    .join("");

  container.innerHTML = `
    <details open>
      <summary class="citations-summary">📚 Grounded Source Passages (${citations.length} reranked chunks)</summary>
      <div class="citations-grid">${cardsHtml}</div>
    </details>
  `;
}

// ==================== HARDWARE & PERMISSIONED SETUP MODAL ====================
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
    document.getElementById("specCpu").textContent = `${hw.cpu_brand} (${hw.cpu_cores_logical} threads)`;
    document.getElementById("specRam").textContent = `${hw.ram_total_gb} GB Total (${hw.ram_available_gb} GB free)`;
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
        <strong>📦 Detected On-Device Models (Reused Automatically — Zero Download Needed):</strong><br/>
        • <strong>Local LLMs:</strong> <code>${escapeHtml(llmList)}</code><br/>
        • <strong>Local Embedders:</strong> <code>${escapeHtml(embList)}</code><br/>
        • <strong>Local Rerankers:</strong> <code>${escapeHtml(rerankList)}</code>
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
    const alts = (tier.llm_alternatives || []).slice(0, 3).join(", ");
    card.innerHTML = `
      <div class="tier-card-top">
        <strong>${escapeHtml(tier.name)}</strong>
        <span class="tier-badge">${escapeHtml(tier.badge)}</span>
      </div>
      <div class="tier-models">LLM: ${escapeHtml(tier.llm_model)} (${tier.llm_size_gb}GB) • Alts: ${escapeHtml(alts)}</div>
      <div class="tier-models">Embed: ${escapeHtml(tier.embedding_model)} • Rerank: ${escapeHtml(tier.reranker_model)}</div>
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
  btn.textContent = "Applying Configuration...";
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
    btn.textContent = "Approve & Apply Recommended Setup";
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

// ==================== THEME & MOBILE SIDEBAR ====================
document.getElementById("themeToggleBtn").addEventListener("click", () => {
  const html = document.documentElement;
  const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
});

document.getElementById("mobileSidebarBtn").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

// ==================== INITIALIZATION ====================
(async function initApp() {
  initSpeechRecognition();
  await refreshStatus();
  // Check if user has profiled hardware yet; if first run, show non-intrusive hardware wizard
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    state.customInstruction = cfg.custom_system_prompt || "";
    customPersonaInput.value = state.customInstruction;
    if (!cfg.hardware_profiled) {
      await loadAndOpenHardwareModal();
    }
  } catch (e) {
    console.error(e);
  }
})();
